"""Scores for one backend's prediction rows.

A row is {"gold", "accepted"?, "pred", "confidence", "latency_ms", "extra"?}. gold is the primary
label; accepted lists every label that counts as correct on a genuinely two-way row.
"""

import math
import random

INVALID = "<invalid>"


def credit(row: dict) -> dict:
    """Score a two-way row as the accepted label the backend picked, else the primary label."""
    accepted = row.get("accepted") or [row["gold"]]
    return {**row, "gold": row["pred"] if row["pred"] in accepted else accepted[0]}


def bootstrap_ci(hits: list[bool], resamples: int = 1000, seed: int = 0) -> tuple[float, float] | None:
    """95% bootstrap interval for the share of hits."""
    if not hits:
        return None
    rng = random.Random(seed)
    n = len(hits)
    means = sorted(sum(rng.choices(hits, k=n)) / n for _ in range(resamples))
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile, q in [0, 100]."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


def per_label(rows: list[dict], labels: list[str]) -> dict[str, dict]:
    out = {}
    for label in labels:
        tp = sum(r["pred"] == label and r["gold"] == label for r in rows)
        fp = sum(r["pred"] == label and r["gold"] != label for r in rows)
        fn = sum(r["gold"] == label and r["pred"] != label for r in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn, "predicted": tp + fp}
    return out


def expected_calibration_error(rows: list[dict], bins: int = 10) -> float | None:
    """|accuracy - confidence| per equal-width confidence bin, weighted by bin size."""
    scored = [r for r in rows if r.get("confidence") is not None]
    if not scored:
        return None
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        in_bin = [r for r in scored if lo < r["confidence"] <= hi or (b == 0 and r["confidence"] == 0)]
        if in_bin:
            acc = sum(r["pred"] == r["gold"] for r in in_bin) / len(in_bin)
            conf = sum(r["confidence"] for r in in_bin) / len(in_bin)
            total += abs(acc - conf) * len(in_bin) / len(scored)
    return total


def confusion(rows: list[dict]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}
    for r in rows:
        pred = r["pred"] if r["pred"] is not None else INVALID
        matrix.setdefault(r["gold"], {}).setdefault(pred, 0)
        matrix[r["gold"]][pred] += 1
    return matrix


def summarize(raw_rows: list[dict], labels: list[str]) -> dict:
    rows = [credit(r) for r in raw_rows]
    n = len(rows)
    hits = [r["pred"] == r["gold"] for r in rows]
    labels_f1 = per_label(rows, labels)
    # Macro-F1 over labels that occur in gold or predictions, as sklearn does.
    active = [label for label, m in labels_f1.items() if m["support"] or m["predicted"]]
    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    summary = {
        "n": n,
        "ambiguous": sum(len(r.get("accepted") or []) > 1 for r in rows),
        "accuracy": sum(hits) / n if n else 0.0,
        "accuracy_ci": bootstrap_ci(hits),
        "macro_f1": sum(labels_f1[label]["f1"] for label in active) / len(active) if active else 0.0,
        "invalid": sum(r["pred"] is None for r in rows),
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "ece": expected_calibration_error(rows),
        "mean_confidence": None,
        "flip_rate": None,
        "first_order_accuracy": None,
        "per_label": labels_f1,
        "confusion": confusion(rows),
    }
    confident = [r["confidence"] for r in rows if r.get("confidence") is not None]
    if confident:
        summary["mean_confidence"] = sum(confident) / len(confident)
    ordered = [r for r in rows if r.get("extra", {}).get("order_choices")]
    if ordered:
        summary["flip_rate"] = sum(len(set(r["extra"]["order_choices"])) > 1 for r in ordered) / len(ordered)
        summary["first_order_accuracy"] = sum(
            r["extra"]["first_order"] in (r.get("accepted") or [r["gold"]]) for r in ordered
        ) / len(ordered)
    return summary
