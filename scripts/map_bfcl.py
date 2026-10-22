"""Map pyuvraj/bfcl-v1-jev-routing onto the tool_router labels, by fixed rules (not hand-checked).

BFCL's gold is "which of these specific tools", ours is a generic next action, so:
- gold tool named get_/find_/search_/locate_/retrieve_/fetch_ (a lookup)         -> web_search
- gold tool that calculates / computes / converts / solves                       -> run_code
- no_tool rows ("the one tool offered doesn't fit") with numbers in the request  -> run_code (v1: math with
  specific numbers is run_code); without numbers                                 -> answer_directly
- anything else (bookings, database edits, analysis services)                    -> dropped

Usage: python scripts/map_bfcl.py data/external/bfcl-v1-jev-routing.jsonl tasks/tool_router_bfcl
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

# A word anywhere in the tool name that means "look something up".
LOOKUP_NAME = re.compile(r"(^|[._])(get|find|finder|search|locate|retrieve|fetch|lookup|forecast|info|details|"
                         r"price|quality|estimate)([._]|$)", re.I)
CALC = re.compile(r"calculat|comput|convert|solv|derivat|integra|factori|probabilit|\barea\b|volume|regression|"
                  r"t_?test|\bmean\b|median|average|interest|\blcm\b|\bgcd\b|roots?\b|speed|velocity|density|growth",
                  re.I)


def map_row(row: dict) -> tuple[str | None, str]:
    if row["expected_route_kind"] != "tool_call":
        return ("run_code", "no_tool + numbers") if re.search(r"\d", row["user_request"]) else ("answer_directly", "no_tool")
    tool = row["gold_action_id"]
    desc = next((o.get("description", "") for o in row["options"] if o["id"] == tool), "")
    if LOOKUP_NAME.search(tool):
        return "web_search", f"lookup tool {tool}"
    if CALC.search(f"{tool} {desc}"):
        return "run_code", f"calc tool {tool}"
    return None, f"dropped {tool}"


def main(src: str, out_dir: str) -> None:
    rows = [json.loads(line) for line in Path(src).read_text(encoding="utf-8").splitlines() if line.strip()]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    task = json.loads(Path("tasks/tool_router/task.json").read_text(encoding="utf-8"))
    task["name"] = "tool_router_bfcl"
    (out / "task.json").write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")

    gold, why = [], Counter()
    for r in rows:
        label, reason = map_row(r)
        why[reason.split(" ")[0] if label else "dropped"] += 1
        if label:
            gold.append({"id": f"bfcl-{r['id']}", "text": r["user_request"], "label": label, "split": "test",
                         "source": "bfcl-mapped", "note": f"{r['stratum']}; {reason}"})
    (out / "gold.jsonl").write_text("".join(json.dumps(g) + "\n" for g in gold), encoding="utf-8")
    print(f"{len(gold)} of {len(rows)} rows mapped -> {out / 'gold.jsonl'}")
    print("labels:", dict(Counter(g["label"] for g in gold)))
    print("why:", dict(why))


if __name__ == "__main__":
    main(*sys.argv[1:3])
