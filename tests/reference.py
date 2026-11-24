"""Reference implementations used to cross-check the production code."""

import itertools
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple


def brute_force_isotonic(
    counts: Dict[int, Tuple[int, int]]
) -> Dict[int, Fraction]:
    """Independent exact isotonic oracle, in O(n^3).

    Uses the min-max characterization of the weighted isotonic regression:
    ``theta_i = max_{a<=i} min_{b>=i} mean(a..b)``.  This shares no code with
    the PAVA implementation under test.
    """
    scores = sorted(counts)
    if not scores:
        return {}
    totals = [counts[s][1] for s in scores]
    positives = [counts[s][0] for s in scores]
    n = len(scores)

    def mean(a: int, b: int) -> Fraction:
        return Fraction(sum(positives[a : b + 1]), sum(totals[a : b + 1]))

    rates: Dict[int, Fraction] = {}
    for i in range(n):
        rates[scores[i]] = max(
            min(mean(a, b) for b in range(i, n)) for a in range(0, i + 1)
        )
    return rates


def reference_rates(
    counts: Dict[int, Tuple[int, int]]
) -> Optional[Dict[int, Fraction]]:
    """Fit plus the left-nearest / lowest fallback, independently written."""
    fitted = brute_force_isotonic(counts)
    if not fitted:
        return None
    inspected = sorted(fitted)
    out: Dict[int, Fraction] = {}
    for score in range(256):
        candidates = [s for s in inspected if s <= score]
        source = candidates[-1] if candidates else inspected[0]
        out[score] = fitted[source]
    return out


def reference_queue(
    panels: Sequence[Tuple[int, int, int, int]],
    rates: Dict[int, Fraction],
    k: int,
) -> List[Tuple[int, Fraction, Fraction]]:
    """Brute-force the top K by scanning every pending board."""
    rows = []
    for panel_id, score, loss, inspected in panels:
        if inspected != -1:
            continue
        rate = rates[score]
        rows.append((panel_id, rate, rate * loss))
    rows.sort(key=lambda r: (-r[2], r[0]))
    return rows[:k]


def reference_audit(
    panels: Sequence[Tuple[int, int, int, int]],
    min_defects: int,
    max_defects: int,
) -> Dict[str, object]:
    """Exhaustive oracle for audit AUC extrema.

    Enumerates every 0/1 assignment of the pending panels, keeps those whose
    total defects lie in the interval and which contain both classes, and
    computes the ROC AUC with half-credit ties.  Returns
    ``{"status": "ok", "min": (auc, p, {score: x}), "max": ...}`` or a
    ``conflict`` / ``incomplete`` status.
    """
    pending = [i for i, (_, _, _, inspected) in enumerate(panels) if inspected == -1]
    known_pos = sum(1 for *_, inspected in panels if inspected == 1)
    n_total = len(panels)

    outcomes = []
    for bits in itertools.product((0, 1), repeat=len(pending)):
        p = sum(bits)
        total_defects = known_pos + p
        if not (min_defects <= total_defects <= max_defects):
            continue
        if total_defects == 0 or total_defects == n_total:
            continue
        labels = [(score, inspected) for _, score, _, inspected in panels]
        for idx, bit in zip(pending, bits):
            labels[idx] = (labels[idx][0], bit)
        concordant = Fraction(0)
        for i, (score_i, y_i) in enumerate(labels):
            for j in range(i):
                score_j, y_j = labels[j]
                if y_i == y_j:
                    continue
                pos_score, neg_score = (
                    (score_i, score_j) if y_i == 1 else (score_j, score_i)
                )
                if pos_score > neg_score:
                    concordant += 1
                elif pos_score == neg_score:
                    concordant += Fraction(1, 2)
        auc = concordant / (total_defects * (n_total - total_defects))
        x_by_score: Dict[int, int] = {}
        for _, score, _, _ in panels:
            x_by_score.setdefault(score, 0)
        for idx, bit in zip(pending, bits):
            score = panels[idx][1]
            x_by_score[score] += bit
        outcomes.append((auc, p, x_by_score))

    if not outcomes:
        interval_feasible = any(
            min_defects <= known_pos + p <= max_defects
            for p in range(len(pending) + 1)
        )
        return {"status": "incomplete" if interval_feasible else "conflict"}
    min_auc = min(row[0] for row in outcomes)
    max_auc = max(row[0] for row in outcomes)
    return {
        "status": "ok",
        "min": (min_auc, min(row[1] for row in outcomes if row[0] == min_auc)),
        "max": (max_auc, min(row[1] for row in outcomes if row[0] == max_auc)),
        "all": outcomes,
    }
