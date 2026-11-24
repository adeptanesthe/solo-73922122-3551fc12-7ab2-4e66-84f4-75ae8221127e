"""Tests for the ROC-AUC audit orders."""

import random
import time
from fractions import Fraction

import pytest
from fastapi.testclient import TestClient

from app.audit import AuditConflict, AuditIncomplete, ScoreBucket, evaluate_audit
from app.main import app, store
from app.store import MAX_PANELS, PanelStore
from tests.reference import reference_audit


def fresh_client() -> TestClient:
    store.__init__()  # type: ignore[misc]
    return TestClient(app)


def buckets_from_store(s: PanelStore):
    return s._score_buckets()


def alloc_by_score(extremum):
    return {row.score: row.as_defect for row in extremum.pending_as_defect}


# ------------------------------------------------------------- math vs oracle
def test_audit_matches_brute_force_random():
    rng = random.Random(20261124)
    for trial in range(250):
        s = PanelStore()
        entries = []
        pid = 0
        known = []
        n_scores = rng.randint(1, 4)
        scores = sorted(rng.sample(range(256), n_scores))
        for score in scores:
            for _ in range(rng.randint(0, 4)):
                pid += 1
                entries.append((pid, score, rng.randint(1, 99), 1))
                known.append((pid, score, 1, 1))
            for _ in range(rng.randint(0, 4)):
                pid += 1
                entries.append((pid, score, rng.randint(1, 99), 0))
                known.append((pid, score, 1, 0))
            for _ in range(rng.randint(0, 4)):
                pid += 1
                entries.append((pid, score, rng.randint(1, 99), None))
                known.append((pid, score, 1, -1))
        if not entries:
            continue
        s.register(entries)
        n_total = len(entries)
        known_pos = sum(1 for *_, y in known if y == 1)
        pending = sum(1 for *_, y in known if y == -1)

        lo = rng.randint(0, n_total)
        hi = rng.randint(lo, n_total)
        order = s.create_audit_order(lo, hi)
        oracle = reference_audit(known, lo, hi)

        if oracle["status"] != "ok":
            with pytest.raises(
                AuditConflict if oracle["status"] == "conflict" else AuditIncomplete
            ):
                s.audit_snapshot()
            continue

        ev = evaluate_audit(buckets_from_store(s), order)
        for kind, extremum in (("min", ev.minimum), ("max", ev.maximum)):
            auc_want, p_want = oracle[kind]
            assert extremum.auc == auc_want, (trial, kind, extremum.auc, auc_want)
            assert extremum.total_defects == known_pos + p_want
            alloc = alloc_by_score(extremum)
            assert sum(alloc.values()) == p_want
            # The concrete allocation must itself attain the extremal AUC.
            matching = [row for row in oracle["all"] if row[0] == auc_want]
            assert any(
                all(alloc.get(score, 0) == x for score, x in row[2].items())
                for row in matching
            )


def test_audit_hand_checked_extremes():
    # score 10: one known good + four pending; score 90: one known defect + four
    # pending.  Interval [1, 3] -> exactly two pending panels are defects.
    s = PanelStore()
    s.register(
        [(1, 10, 1, 0), (2, 90, 1, 1)]
        + [(100 + i, 10, 1, None) for i in range(4)]
        + [(200 + i, 90, 1, None) for i in range(4)]
    )
    order = s.create_audit_order(1, 3)
    ev = s.audit_snapshot()
    # Min: both pending defects placed at the low score -> AUC 8/21.
    assert ev.minimum.auc == Fraction(8, 21)
    assert alloc_by_score(ev.minimum) == {10: 2, 90: 0}
    # Max: both placed at the high score -> AUC 6/7.
    assert ev.maximum.auc == Fraction(6, 7)
    assert alloc_by_score(ev.maximum) == {10: 0, 90: 2}
    assert ev.feasible_defect_range == (1, 3)
    assert ev.known_defects == 1


def test_audit_ties_count_half():
    # One known positive and one known negative at the SAME score: every
    # comparison ties, so AUC must be exactly 1/2.
    s = PanelStore()
    s.register([(1, 50, 1, 1), (2, 50, 1, 0)])
    order = s.create_audit_order(0, 2)
    ev = evaluate_audit(s._score_buckets(), order)
    assert ev.minimum.auc == Fraction(1, 2)
    assert ev.maximum.auc == Fraction(1, 2)


def test_audit_conflict_when_known_defects_exceed_interval():
    s = PanelStore()
    s.register([(1, 10, 1, 1), (2, 20, 1, 1), (3, 30, 1, None)])
    s.create_audit_order(0, 1)  # two known defects cannot fit
    with pytest.raises(AuditConflict):
        s.audit_snapshot()


def test_audit_incomplete_when_a_class_must_be_empty():
    # All panels are inspected as defects: no negative class exists under any
    # assignment compatible with the forced interval.
    s = PanelStore()
    s.register([(1, 10, 1, 1), (2, 20, 1, 1)])
    s.create_audit_order(2, 2)
    with pytest.raises(AuditIncomplete):
        s.audit_snapshot()
    # A pending panel plus an interval permitting a good label restores it.
    s.register([(3, 30, 1, None)])
    s.create_audit_order(1, 2)
    ev = s.audit_snapshot()
    assert ev.maximum.total_defects >= 1


def test_audit_incomplete_when_positive_class_cannot_exist():
    # Known-clean panels only and an interval pinned at zero: no positive class.
    s = PanelStore()
    s.register([(1, 10, 1, 0), (2, 20, 1, None)])
    s.create_audit_order(0, 0)
    with pytest.raises(AuditIncomplete):
        s.audit_snapshot()


def test_audit_correction_takes_effect_and_recovers_query():
    # Interval [0, 1] with one known defect: queryable.  A second confirmed
    # defect creates a conflict; correcting it back restores the same order.
    s = PanelStore()
    s.register([(1, 10, 1, 1), (2, 20, 1, 0), (3, 30, 1, None)])
    order = s.create_audit_order(0, 1)
    ev = s.audit_snapshot()
    assert ev.minimum.auc is not None

    s.set_inspection(3, 1)
    with pytest.raises(AuditConflict):
        s.audit_snapshot()
    # The order is retained untouched.
    assert s.audit_order.order_id == order.order_id

    s.set_inspection(3, 0)
    recovered = s.audit_snapshot()
    assert recovered.minimum.auc == ev.minimum.auc
    assert recovered.maximum.auc == ev.maximum.auc


def test_audit_pending_fill_in_changes_extrema_without_touching_heaps():
    s = PanelStore()
    s.register([(i, i % 256, i + 1, None) for i in range(20_000)])
    s.register([(90_000, 10, 1, 0), (90_001, 200, 1, 1)])
    s.create_audit_order(5_000, 9_000)
    before = s.audit_snapshot()
    s.reset_stats()
    # A single-panel correction updates only counters; no pending heap scan.
    s.set_inspection(7, 1)
    assert s.stats == {"heap_entries_dropped": 0, "heap_entries_popped": 0}
    after = s.audit_snapshot()
    assert after.inspected_panels == before.inspected_panels + 1
    assert after.pending_panels == before.pending_panels - 1
    assert after.known_defects == before.known_defects + 1


def test_score_bucket_totals_match_store_throughout_mutations():
    """The audit's per-score aggregates must never drift from the panel store."""
    rng = random.Random(321)
    s = PanelStore()
    entries = [
        (i, rng.randrange(256), rng.randint(1, 9), rng.choice([None, None, 0, 1]))
        for i in range(3_000)
    ]
    s.register(entries)
    for _ in range(400):
        pid = rng.randrange(3_000)
        if s.get_panel(pid).is_inspected:
            continue
        s.set_inspection(pid, rng.choice((0, 1)))
        buckets = {b.score: b for b in s._score_buckets()}
        # Pending totals from the 256 counters equal the store-level count.
        assert sum(b.pending for b in buckets.values()) == s.pending_panels
        assert sum(b.total for b in buckets.values()) == s.total_panels
        assert (
            sum(b.inspected_defects + b.inspected_good for b in buckets.values())
            == s.inspected_panels
        )


def test_audit_query_reads_only_score_buckets():
    # The evaluation input is at most 256 aggregate rows even at the cap.
    s = PanelStore()
    s.register([(i, i % 256, (i % 50) + 1, None) for i in range(MAX_PANELS - 4)])
    s.register(
        [
            (MAX_PANELS + 1, 10, 1, 0),
            (MAX_PANELS + 2, 10, 1, 1),
            (MAX_PANELS + 3, 200, 1, 0),
            (MAX_PANELS + 4, 200, 1, 1),
        ]
    )
    s.create_audit_order(10_000, 40_000)
    started = time.perf_counter()
    ev = s.audit_snapshot()
    elapsed = time.perf_counter() - started
    assert len(ev.score_buckets) <= 256
    assert ev.total_panels == MAX_PANELS
    # No label assignment is enumerated; the sweep is linear in pending count.
    assert elapsed < 5.0
    assert 0.0 <= ev.minimum.auc <= ev.maximum.auc <= 1.0


# ----------------------------------------------------------------------- API
def test_api_audit_lifecycle_conflict_and_recovery():
    client = fresh_client()
    client.post(
        "/panels",
        json={
            "panels": [
                {"panel_id": 1, "score": 10, "loss": 1, "inspection_result": 1},
                {"panel_id": 2, "score": 20, "loss": 1, "inspection_result": 0},
                {"panel_id": 3, "score": 30, "loss": 1},
                {"panel_id": 4, "score": 40, "loss": 1},
            ]
        },
    )

    created = client.post("/audits", json={"min_defects": 1, "max_defects": 1})
    assert created.status_code == 201
    order_id = created.json()["order_id"]
    assert created.json()["panels_at_creation"] == 4

    result = client.get(f"/audits/{order_id}").json()
    assert result["order"]["order_id"] == order_id
    assert result["known_defects"] == 1
    assert result["feasible_defect_range"] == [1, 1]
    assert result["minimum"]["auc"] <= result["maximum"]["auc"]
    # Allocation rows only mention scores that have pending panels.
    mentioned = {
        row["score"]
        for ext in (result["minimum"], result["maximum"])
        for row in ext["pending_as_defect"]
    }
    assert mentioned == {30, 40}
    assert len(result["score_buckets"]) == 4

    # /audits/current returns the same evaluation.
    assert client.get("/audits/current").json()["minimum"] == result["minimum"]

    # A second confirmed defect conflicts with the interval.
    client.put("/panels/3/inspection", json={"inspection_result": 1})
    conflict = client.get("/audits/current")
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "audit_conflict"

    # Correcting it restores the original order's query.
    client.put("/panels/3/inspection", json={"inspection_result": 0})
    restored = client.get("/audits/current")
    assert restored.status_code == 200
    assert restored.json()["minimum"]["auc"] == result["minimum"]["auc"]


def test_api_audit_incomplete_returns_reason_only():
    client = fresh_client()
    client.post(
        "/panels",
        json={
            "panels": [
                {"panel_id": 1, "score": 10, "loss": 1, "inspection_result": 1},
                {"panel_id": 2, "score": 20, "loss": 1, "inspection_result": 1},
            ]
        },
    )
    client.post("/audits", json={"min_defects": 2, "max_defects": 2})
    response = client.get("/audits/current")
    assert response.status_code == 409
    body = response.json()
    assert body["code"] == "audit_incomplete"
    assert "both" in body["detail"]


def test_api_creating_order_replaces_current():
    client = fresh_client()
    client.post(
        "/panels",
        json={"panels": [
            {"panel_id": 1, "score": 10, "loss": 1, "inspection_result": 0},
            {"panel_id": 2, "score": 20, "loss": 1},
        ]},
    )
    first = client.post("/audits", json={"min_defects": 0, "max_defects": 1}).json()
    second = client.post("/audits", json={"min_defects": 1, "max_defects": 1}).json()
    assert first["order_id"] != second["order_id"]
    # The old id is no longer resolvable; only the current order is kept.
    old = client.get(f"/audits/{first['order_id']}")
    assert old.status_code == 404
    assert old.json()["code"] == "not_found"
    current = client.get("/audits/current").json()
    assert current["order"]["order_id"] == second["order_id"]
    assert client.get("/health").json()["audit_order_id"] == second["order_id"]


def test_api_audit_validation_errors():
    client = fresh_client()
    # No panels registered.
    assert client.post("/audits", json={"min_defects": 0, "max_defects": 1}).status_code == 400
    client.post(
        "/panels",
        json={"panels": [{"panel_id": 1, "score": 10, "loss": 1}]},
    )
    # Inverted interval.
    assert client.post("/audits", json={"min_defects": 2, "max_defects": 1}).status_code == 422
    # Negative bound.
    assert client.post("/audits", json={"min_defects": -1, "max_defects": 1}).status_code == 422
    # Bound beyond the registered population.
    assert client.post("/audits", json={"min_defects": 5, "max_defects": 9}).status_code == 400
    # max_defects beyond the population is rejected as well.
    assert client.post("/audits", json={"min_defects": 0, "max_defects": 9}).status_code == 400
    # Extra field rejected.
    assert client.post(
        "/audits", json={"min_defects": 0, "max_defects": 1, "extra": 1}
    ).status_code == 422
    # No order was created by the rejected requests.
    assert client.get("/audits/current").status_code == 404


def test_audit_with_no_pending_panels_uses_fixed_labels():
    # Everything inspected: the only feasible p is zero and the AUC is fixed.
    s = PanelStore()
    s.register(
        [(1, 10, 1, 0), (2, 20, 1, 0), (3, 30, 1, 1), (4, 40, 1, 1)]
    )
    s.create_audit_order(2, 2)
    ev = s.audit_snapshot()
    assert ev.pending_panels == 0
    assert ev.minimum.auc == ev.maximum.auc
    # Every positive outranks every negative: AUC = 1.
    assert ev.minimum.auc == Fraction(1)
    assert all(row.as_defect == 0 for row in ev.minimum.pending_as_defect)


def test_audit_max_defects_below_known_defects_conflicts():
    s = PanelStore()
    s.register([(1, 10, 1, 1), (2, 20, 1, 1), (3, 30, 1, None)])
    s.create_audit_order(0, 1)
    with pytest.raises(AuditConflict):
        s.audit_snapshot()
    # Widening the same order family resolves it.
    s.create_audit_order(0, 3)
    ev = s.audit_snapshot()
    assert ev.feasible_defect_range == (2, 3)


def test_api_audit_rendered_allocations_sum_to_defects():
    client = fresh_client()
    panels = (
        [
            {"panel_id": 1, "score": 10, "loss": 1, "inspection_result": 1},
            {"panel_id": 2, "score": 40, "loss": 1, "inspection_result": 0},
            {"panel_id": 99, "score": 90, "loss": 1, "inspection_result": 0},
        ]
        + [{"panel_id": 100 + i, "score": 10, "loss": 1} for i in range(6)]
        + [{"panel_id": 200 + i, "score": 200, "loss": 1} for i in range(6)]
    )
    client.post("/panels", json={"panels": panels})
    # Known defects = 1, pending = 12 -> feasible totals inside [3, 12].
    client.post("/audits", json={"min_defects": 3, "max_defects": 12})
    response = client.get("/audits/current")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["known_defects"] == 1
    for ext in (body["minimum"], body["maximum"]):
        pending_as_defect = sum(r["as_defect"] for r in ext["pending_as_defect"])
        assert ext["total_defects"] == body["known_defects"] + pending_as_defect
        assert body["feasible_defect_range"][0] <= ext["total_defects"]
        assert ext["total_defects"] <= body["feasible_defect_range"][1]
