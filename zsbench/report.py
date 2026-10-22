"""Turn a run directory (<backend>.jsonl + <backend>.meta.json) into a markdown comparison."""

import json
from pathlib import Path

from .metrics import INVALID, summarize

BACKEND_ORDER = ["rules", "jev", "llm"]


def load_run(run_dir: Path) -> dict[str, tuple[dict, list[dict]]]:
    runs = {}
    for meta_path in sorted(run_dir.glob("*.meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rows_path = run_dir / f"{meta['backend']}.jsonl"
        rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        runs[meta["backend"]] = (meta, rows)
    ordered = sorted(runs, key=lambda b: BACKEND_ORDER.index(b) if b in BACKEND_ORDER else len(BACKEND_ORDER))
    return {b: runs[b] for b in ordered}


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:.1f}%"


def _ms(x: float | None) -> str:
    return "-" if x is None else f"{x:.0f}"


def _num(x: float | None) -> str:
    return "-" if x is None else f"{x:.3f}"


def _acc(s: dict) -> str:
    ci = s["accuracy_ci"]
    if ci is None:
        return _pct(s["accuracy"])
    return f"{_pct(s['accuracy'])} ({ci[0] * 100:.0f}-{ci[1] * 100:.0f})"


def _split(rows: list[dict], split: str) -> list[dict]:
    return [r for r in rows if r.get("split") == split]


def render(run_dir: str | Path) -> tuple[str, str]:
    """Return (headline table, full report).

    When gold has a test split, headline numbers and per-label F1 come from test only, while the
    confusion matrices and the list of misses come from dev only: tune on dev, report test, and never
    look at individual test errors."""
    run_dir = Path(run_dir)
    runs = load_run(run_dir)
    if not runs:
        raise FileNotFoundError(f"no *.meta.json in {run_dir}")
    labels = next(iter(runs.values()))[0]["labels"]
    all_rows = [r for _, rows in runs.values() for r in rows]
    has_test = any(r.get("split") == "test" for r in all_rows)
    has_dev = any(r.get("split") == "dev" for r in all_rows)
    only_dev = has_dev and all(r.get("split") == "dev" for r in all_rows)
    scope = "test split" if has_test else "dev split" if only_dev else "all rows"
    summaries = {b: summarize(_split(rows, "test") if has_test else rows, labels) for b, (_, rows) in runs.items()}
    dev_summaries = {b: summarize(_split(rows, "dev"), labels) for b, (_, rows) in runs.items()} if has_dev else {}

    head = [
        f"Scope: {scope}",
        "",
        "| backend | model | n | accuracy (95% CI) | macro-F1 | invalid | p50 ms | p95 ms | mean conf | ECE | order flips | acc, fixed order |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for b, s in summaries.items():
        meta = runs[b][0]
        head.append(
            f"| {b} | {meta.get('model', '-')} | {s['n']} | {_acc(s)} | {_num(s['macro_f1'])} | "
            f"{s['invalid']} | {_ms(s['p50_ms'])} | {_ms(s['p95_ms'])} | {_pct(s['mean_confidence'])} | "
            f"{_num(s['ece'])} | {_pct(s['flip_rate'])} | {_pct(s['first_order_accuracy'])} |"
        )
    headline = "\n".join(head)

    first_meta = next(iter(runs.values()))[0]
    first = next(iter(summaries.values()))
    lines = [
        f"# Run report: {run_dir.name}",
        "",
        f"Task: `{first_meta['task']}`, split: {first_meta.get('split') or 'all'}, "
        f"two-way rows (either label counts): {first['ambiguous']} of {first['n']}",
        "",
        headline,
        "",
    ]
    lines += [
        "- **accuracy** uses the debiased prediction (Jev: averaged over label orders). "
        "The 95% CI is a bootstrap over rows; gaps between backends smaller than it are noise.",
        "- **acc, fixed order** is Jev asked once with labels in task.json order, i.e. a naive single call.",
        "- **order flips** is the share of inputs where Jev's choice changed with label order (position bias).",
        "- **ECE** compares stated confidence with real accuracy; lower is better. Simple Jev's probabilities are not calibrated.",
        "",
        f"## F1 per label ({scope})",
        "",
        "A rough guide when support is under about 30: one or two mistakes move these a lot.",
        "",
        "| label | support | " + " | ".join(summaries) + " |",
        "|---|---|" + "---|" * len(summaries),
    ]
    for label in labels:
        cells = " | ".join(_num(s["per_label"][label]["f1"]) for s in summaries.values())
        lines.append(f"| {label} | {first['per_label'][label]['support']} | {cells} |")

    if has_test and has_dev:
        lines += ["", "## By split", "", "| backend | dev (tune here) | test (report this) |", "|---|---|---|"]
        for b in summaries:
            d, t = dev_summaries[b], summaries[b]
            lines.append(f"| {b} | {_acc(d)}, n={d['n']} | {_acc(t)}, n={t['n']} |")

    confusion_scope = "dev only" if has_dev else "all rows"
    if has_test and not has_dev:
        lines += ["", "Confusion matrices and misses are hidden: this run has only test rows. "
                      "Run with the dev split to see errors you can tune on."]
    for b, s in ({} if has_test and not has_dev else dev_summaries or summaries).items():
        cols = labels + ([INVALID] if s["invalid"] else [])
        lines += ["", f"## Confusion: {b} ({confusion_scope}; rows = gold, cols = predicted)", ""]
        lines.append("| gold \\ pred | " + " | ".join(cols) + " |")
        lines.append("|---|" + "---|" * len(cols))
        for gold in labels:
            row = s["confusion"].get(gold, {})
            lines.append(f"| {gold} | " + " | ".join(str(row.get(c, 0)) for c in cols) + " |")

    if has_dev:
        lines += ["", "## Dev misses (tune wording on these)", "",
                  "Test misses are left out on purpose: looking at them turns test into another dev set.", ""]
        for b, (_, rows) in runs.items():
            misses = [r for r in _split(rows, "dev") if r["pred"] not in (r.get("accepted") or [r["gold"]])]
            lines += [f"### {b}: {len(misses)} misses", ""]
            if misses:
                lines += ["| id | gold | predicted | text |", "|---|---|---|---|"]
                for r in misses:
                    text = (r.get("text") or "").replace("|", "/").replace("\n", " ")[:100]
                    lines.append(f"| {r['id']} | {r['gold']} | {r['pred'] or INVALID} | {text} |")
                lines.append("")

    return headline, "\n".join(lines) + "\n"
