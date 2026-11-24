"""ROC-AUC extrema for the score-audit order.

The audit asks whether the raw defect score ranks defective panels ahead of
good ones.  Only a batch-wide defect interval ``[min_defects, max_defects]`` is
known downstream and some panels still lack an inspection result, so the audit
reports the minimum and maximum ROC AUC attainable by *any* label assignment
that:

* keeps every recorded inspection result,
* puts the total number of defects inside ``[min_defects, max_defects]``, and
* leaves both classes non-empty (an AUC is undefined otherwise).

Ties (a positive and a negative sharing a score) count half a concordant pair.

Everything is derived from per-score aggregate counts -- individual panels are
never enumerated -- and all arithmetic is exact (integers /
:class:`fractions.Fraction`) until the response is rendered.

Derivation
----------
For score ``s`` let ``a_s`` be known defects, ``b_s`` known good panels and
``u_s`` pending panels; ``x_s`` pending panels labelled defective, with
``0 <= x_s <= u_s`` and ``sum x_s = p``.  With ``A = sum a_s`` and
``N = sum (a_s + b_s + u_s)`` the number of concordant (weighted, half-credit)
pairs is

    T(x) = T0 + sum_s u_s d_s + sum_s x_s c_s - p(p-1)/2,

where ``T0`` counts known-positive/known-negative pairs, ``d_s`` is the known
defect contribution a pending panel at score ``s`` is compared against (known
positives above s, plus half of those at s), and ``c_s`` is the constant
marginal concordant-pair gain of labelling one more pending panel of score
``s`` defective:

    c_s = (known negatives below s) + (known negatives at s)/2
        + (pending panels below s) + (u_s - 1)/2 - d_s.

The two half-credit terms are, respectively, the same-score tie against a
known negative and against another pending panel; ``-p(p-1)/2`` then removes
the pending-positive/pending-negative pairs that were counted twice (once from
each endpoint), a correction that is global in ``p`` because every such pair
spans two labelled pending panels.

For a *fixed* ``p`` the ``p(p-1)/2`` term is constant and the remaining
allocation problem is a bounded unit-item knapsack with constant item weight
``c_s`` per bucket: fill pending slots in descending ``c_s`` order to maximise
the concordant-pair count, ascending to minimise it.  The AUC denominator
``(A+p)(N-A-p)`` changes with p, so every feasible p is solved independently
and the smallest/largest AUC is retained.  Only the at-most-256 score buckets
are ever visited; no panel-level label assignment is enumerated.
"""

import bisect
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
from typing import List, Optional, Tuple

SCORE_COUNT = 256


class AuditError(LookupError):
    """Base class for audit queries that cannot produce an AUC."""


class AuditConflict(AuditError):
    """Recorded inspections already contradict the defect-count interval."""


class AuditIncomplete(AuditError):
    """Every feasible assignment misses a positive or a negative class."""


@dataclass(frozen=True)
class AuditOrder:
    """A created audit order (the defect interval is the only payload)."""

    order_id: str
    min_defects: int
    max_defects: int
    created_at: str
    panels_at_creation: int
    inspected_at_creation: int
    pending_at_creation: int


@dataclass(frozen=True)
class ScoreBucket:
    """Per-score aggregate counts; the finest granularity the audit reads."""

    score: int
    total: int
    inspected_defects: int  # a_s
    inspected_good: int  # b_s
    pending: int  # u_s


@dataclass(frozen=True)
class PendingAsDefect:
    """How many pending panels of a score are treated as defective."""

    score: int
    pending: int
    as_defect: int


@dataclass(frozen=True)
class Extremum:
    auc: Fraction
    total_defects: int
    pending_as_defect: Tuple[PendingAsDefect, ...]


@dataclass(frozen=True)
class AuditEvaluation:
    order: AuditOrder
    total_panels: int
    inspected_panels: int
    pending_panels: int
    known_defects: int
    feasible_defect_range: Tuple[int, int]
    minimum: Extremum
    maximum: Extremum
    score_buckets: Tuple[ScoreBucket, ...] = field(default_factory=tuple)


def _extremum(
    descending: bool,
    *,
    m: int,
    u: List[int],
    c: List[Fraction],
    q_lo: int,
    q_hi: int,
    a_sum: int,
    n_total: int,
    t_base: Fraction,
    pending_rows: List[ScoreBucket],
) -> Extremum:
    """Extremal AUC over the feasible pending-defect counts p.

    For a fixed p the allocation-dependent part of the concordant-pair count
    is the linear program ``min/max sum x_s c_s`` subject to
    ``0 <= x_s <= u_s`` and ``sum x_s = p``; it is solved by filling buckets
    in ascending (min) or descending (max) order of ``c_s``.  Those bucket
    prefix sums are precomputed once, so every feasible p is answered with a
    binary search: the whole sweep is O(m log m + range · log m) time and
    never enumerates panel-level assignments.
    """
    order_idx = sorted(range(m), key=lambda i: c[i], reverse=descending)
    # Prefix capacity and prefix weighted capacity over the sorted buckets.
    cap_prefix = [0]
    weight_prefix = [Fraction(0)]
    for i in order_idx:
        cap_prefix.append(cap_prefix[-1] + u[i])
        weight_prefix.append(weight_prefix[-1] + u[i] * c[i])

    best_auc: Optional[Fraction] = None
    best_p = q_lo
    best_cut = 0
    best_partial = 0

    for p in range(q_lo, q_hi + 1):
        # Number of sorted buckets the greedy fill consumes entirely: the
        # largest prefix whose capacity is <= p.  bisect_right keeps an exact
        # capacity hit on the full side instead of starting an empty partial.
        full = bisect.bisect_right(cap_prefix, p) - 1
        partial = p - cap_prefix[full]
        if full >= 0:
            weighted = weight_prefix[full]
            if partial:
                weighted += partial * c[order_idx[full]]
        else:
            weighted = Fraction(0)
        concordant = t_base + weighted - Fraction(p * (p - 1), 2)
        positives = a_sum + p
        negatives = n_total - a_sum - p
        auc = concordant / (positives * negatives)
        if best_auc is None or (auc > best_auc if descending else auc < best_auc):
            best_auc = auc
            best_p = p
            best_cut = full
            best_partial = partial

    # Materialise the winning allocation once.
    best_x = [0] * m
    for j in range(best_cut):
        i = order_idx[j]
        best_x[i] = u[i]
    if best_partial:
        best_x[order_idx[best_cut]] = best_partial
    best_x = tuple(best_x)

    assert best_auc is not None and best_x is not None
    rows = tuple(
        PendingAsDefect(
            score=bucket.score,
            pending=bucket.pending,
            as_defect=best_x[local],
        )
        for local, bucket in enumerate(pending_rows)
        if bucket.pending > 0
    )
    return Extremum(
        auc=best_auc,
        total_defects=a_sum + best_p,
        pending_as_defect=rows,
    )


def evaluate_audit(rows: List[ScoreBucket], order: AuditOrder) -> AuditEvaluation:
    """Compute both AUC extrema from score-bucket statistics.

    Raises :class:`AuditConflict` when the known defects cannot fit the order's
    interval, or :class:`AuditIncomplete` when no feasible assignment contains
    both a positive and a negative class.
    """
    rows = sorted(rows, key=lambda bucket: bucket.score)
    m = len(rows)
    a = [row.inspected_defects for row in rows]
    b = [row.inspected_good for row in rows]
    u = [row.pending for row in rows]

    n_total = sum(a) + sum(b) + sum(u)
    a_sum = sum(a)
    u_sum = sum(u)
    inspected = sum(a) + sum(b)

    # Feasible number p of pending panels labelled defective.
    p_lo = max(0, order.min_defects - a_sum)
    p_hi = min(u_sum, order.max_defects - a_sum)
    if p_lo > p_hi:
        raise AuditConflict(
            f"known defects ({a_sum}) cannot fit the ordered interval "
            f"[{order.min_defects}, {order.max_defects}] with {u_sum} "
            "pending panels"
        )
    # Both classes must be present for at least one p in the feasible range:
    # positives A+p >= 1 and negatives N-A-p >= 1.
    q_lo = max(p_lo, 1 - a_sum)
    q_hi = min(p_hi, n_total - a_sum - 1)
    if q_lo > q_hi:
        raise AuditIncomplete(
            "no feasible assignment contains both defective and good panels "
            f"under the interval [{order.min_defects}, {order.max_defects}]"
        )

    # T0: concordant pairs between known positives and known negatives.
    t0 = Fraction(0)
    b_below = 0
    for s in range(m):
        t0 += a[s] * (b_below + Fraction(b[s], 2))
        b_below += b[s]

    # d_s: known positives strictly above s plus half of those at s.
    d: List[Fraction] = [Fraction(0)] * m
    a_above = a_sum
    for s in range(m):
        a_above -= a[s]
        d[s] = a_above + Fraction(a[s], 2)

    t_base = t0 + sum(Fraction(u[s]) * d[s] for s in range(m))

    # c_s: constant per-slot weight in the fixed-p bounded knapsack; the
    # p(p-1)/2 self correction is applied globally once p is chosen.
    b_below = 0
    u_below = 0
    c: List[Fraction] = [Fraction(0)] * m
    for s in range(m):
        e = b_below + Fraction(b[s], 2)
        f = u_below + Fraction(u[s] - 1, 2)
        c[s] = e + f - d[s]
        b_below += b[s]
        u_below += u[s]

    common = dict(
        m=m,
        u=u,
        c=c,
        q_lo=q_lo,
        q_hi=q_hi,
        a_sum=a_sum,
        n_total=n_total,
        t_base=t_base,
        pending_rows=rows,
    )
    minimum = _extremum(False, **common)
    maximum = _extremum(True, **common)

    return AuditEvaluation(
        order=order,
        total_panels=n_total,
        inspected_panels=inspected,
        pending_panels=u_sum,
        known_defects=a_sum,
        feasible_defect_range=(a_sum + p_lo, a_sum + p_hi),
        minimum=minimum,
        maximum=maximum,
        score_buckets=tuple(rows),
    )


def new_order(
    order_id: str,
    min_defects: int,
    max_defects: int,
    *,
    panels: int,
    inspected: int,
    pending: int,
) -> AuditOrder:
    return AuditOrder(
        order_id=order_id,
        min_defects=min_defects,
        max_defects=max_defects,
        created_at=datetime.now(timezone.utc).isoformat(),
        panels_at_creation=panels,
        inspected_at_creation=inspected,
        pending_at_creation=pending,
    )
