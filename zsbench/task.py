"""A task is one agent decision: a question, a label set, naive rules, and a gold set."""

import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

SPLITS = ("dev", "test")


@dataclass
class Task:
    name: str
    question: str
    labels: dict[str, str]  # label -> description shown to the models
    rules: list[dict]  # [{"label": ..., "patterns": [regex, ...]}], first match wins
    default_label: str
    gold: list[dict]  # [{"id", "text", "label", "context"?, "split"?, "source"?, "note"?}]
    path: Path

    @property
    def gold_path(self) -> Path:
        return self.path / "gold.jsonl"


def accepted_labels(row: dict) -> list[str]:
    """A gold label is one label, or a list of labels that are all correct for a genuinely two-way row."""
    label = row["label"]
    return list(label) if isinstance(label, list) else [label]


def context_turns(row: dict) -> list[str]:
    context = row.get("context") or []
    return [context] if isinstance(context, str) else list(context)


def row_key(row: dict) -> str:
    """What makes two rows the same input: the earlier turns plus the text."""
    return "\n".join(context_turns(row) + [row["text"]])


def load_task(task_dir: str | Path) -> Task:
    task_dir = Path(task_dir)
    cfg = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    return Task(
        name=cfg["name"],
        question=cfg["question"],
        labels=cfg["labels"],
        rules=cfg.get("rules", []),
        default_label=cfg.get("default_label", next(iter(cfg["labels"]))),
        gold=read_jsonl(task_dir / "gold.jsonl"),
        path=task_dir,
    )


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate(task: Task) -> list[str]:
    problems = []
    if len(task.labels) < 2:
        problems.append("need at least 2 labels")
    if task.default_label not in task.labels:
        problems.append(f"default_label {task.default_label!r} is not a label")
    for rule in task.rules:
        if rule.get("label") not in task.labels:
            problems.append(f"rule label {rule.get('label')!r} is not a label")
    seen = set()
    for n, row in enumerate(task.gold, 1):
        missing = [k for k in ("id", "text", "label") if k not in row]
        if missing:
            problems.append(f"gold row {n}: missing {', '.join(missing)}")
            continue
        accepted = accepted_labels(row)
        if not accepted or not all(isinstance(label, str) and label in task.labels for label in accepted):
            problems.append(f"gold row {n}: unknown or empty label {row['label']!r}")
        if row["id"] in seen:
            problems.append(f"gold row {n}: duplicate id {row['id']!r}")
        seen.add(row["id"])
        if row.get("split") not in (None, *SPLITS):
            problems.append(f"gold row {n}: split must be one of {SPLITS}, got {row['split']!r}")
        if not all(isinstance(turn, str) for turn in context_turns(row)):
            problems.append(f"gold row {n}: context must be a string or a list of strings")
    return problems


def gold_groups(task: Task) -> dict[str, str]:
    """Gold row id -> group id. Rows are joined when they came from one generated group (contrastive
    pairs, via candidates.jsonl) or are near-copies of each other, so a split never separates them."""
    rows = task.gold
    candidate_group = {c["id"]: c["group"] for c in read_jsonl(task.path / "candidates.jsonl")}
    parent = {r["id"]: r["id"] for r in rows}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    first_in_group: dict[str, str] = {}
    for r in rows:
        group = candidate_group.get(r["id"])
        if group is not None:
            union(r["id"], first_in_group.setdefault(group, r["id"]))
    for i, j in near_copies([row_key(r) for r in rows]):
        union(rows[i]["id"], rows[j]["id"])
    return {r["id"]: find(r["id"]) for r in rows}


def assign_splits(task: Task, dev_frac: float = 1 / 3, seed: int = 0) -> Counter:
    """Tag untagged gold rows dev or test, in place. Whole groups move together, existing tags never
    change, and each label keeps about `dev_frac` of its rows in dev as the set grows."""
    group_of = gold_groups(task)
    groups: dict[str, list[dict]] = {}
    for row in task.gold:
        groups.setdefault(group_of[row["id"]], []).append(row)

    total, dev = Counter(), Counter()
    for row in task.gold:
        label = accepted_labels(row)[0]
        total[label] += 1
        dev[label] += row.get("split") == "dev"

    pending = []
    for members in groups.values():
        tagged = {r["split"] for r in members if r.get("split") in SPLITS}
        if tagged:  # a group already (partly) tagged keeps one split for all its rows
            split = "dev" if "dev" in tagged else "test"
            for r in members:
                if r.get("split") != split:
                    dev[accepted_labels(r)[0]] += (split == "dev") - (r.get("split") == "dev")
                    r["split"] = split
        else:
            pending.append(members)

    random.Random(seed).shuffle(pending)
    for members in pending:
        label = accepted_labels(members[0])[0]
        split = "dev" if dev[label] < dev_frac * total[label] else "test"
        for r in members:
            r["split"] = split
            dev[accepted_labels(r)[0]] += split == "dev"
    return Counter(r["split"] for r in task.gold)


def write_gold(task: Task) -> None:
    task.gold_path.write_text("".join(json.dumps(r) + "\n" for r in task.gold), encoding="utf-8")


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def near_copies(texts: list[str], others: list[str] | None = None, jaccard: float = 0.8) -> list[tuple[int, int]]:
    """Index pairs that are the same after normalizing case and punctuation, or, for 4+ word
    texts, share at least `jaccard` of their words. With others=None, finds pairs within texts."""
    a = [normalize(t) for t in texts]
    b = a if others is None else [normalize(t) for t in others]
    words_a = [set(t.split()) for t in a]
    words_b = words_a if others is None else [set(t.split()) for t in b]
    pairs = []
    for i, (norm, words) in enumerate(zip(a, words_a)):
        for j in range(i + 1 if others is None else 0, len(b)):
            other = words_b[j]
            if norm == b[j] or (
                min(len(words), len(other)) >= 4 and len(words & other) / len(words | other) >= jaccard
            ):
                pairs.append((i, j))
    return pairs
