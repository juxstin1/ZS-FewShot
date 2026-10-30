"""CLI: python -m zsbench {check,split,prompts,run,report,label,generate,blindcheck,adjudicate,route} ..."""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from .backends import AnswersBackend, BackendError, JevBackend, LLMBackend, RulesBackend, chat_prompt
from .env import ENV_FILE, load_dotenv
from .report import render
from .router import Router
from .synth import (
    KINDS,
    ClaudeAdjudicator,
    GenerationError,
    adjudicate,
    blindcheck,
    generate,
    make_generator,
    take_whole_groups,
)
from .task import (
    SPLITS,
    Task,
    accepted_labels,
    assign_splits,
    context_turns,
    load_task,
    near_copies,
    read_jsonl,
    row_key,
    validate,
    write_gold,
)

GOLD_TARGET = 200
MAX_TWO_WAY_SHARE = 0.10
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_GENERATOR = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
DEFAULT_CHECKERS = "openai/gpt-oss-120b,XiaomiMiMo/MiMo-V2.6-Flash"
DEFAULT_TIEBREAKER = "meta-llama/Llama-4-Scout-17B-16E-Instruct"


def parse_named(specs: list[str], flag: str = "--jev-server", what: str = "URL") -> dict[str, tuple[str, str]]:
    """NAME=URL[,MODEL] (or NAME=FILE[,MODEL]) -> {name: (url, model)}; the model name defaults to NAME."""
    servers = {}
    for spec in specs:
        name, sep, rest = spec.partition("=")
        if not sep or not name or not rest:
            raise SystemExit(f"{flag} wants NAME={what}[,MODEL], got {spec!r}")
        url, _, model = rest.partition(",")
        servers[name.strip()] = (url.strip(), model.strip() or name.strip())
    return servers


def make_backend(name: str, task: Task, args: argparse.Namespace):
    servers = parse_named(args.jev_server)
    answers = parse_named(args.answers, "--answers", "FILE")
    if name in answers:
        path, model = answers[name]
        return AnswersBackend(task, path, model, name)
    if name == "rules":
        return RulesBackend(task)
    if name == "jev":
        return JevBackend(task, args.jev_url, args.jev_model, orders=args.orders, timeout=args.timeout)
    if name in servers:
        url, model = servers[name]
        return JevBackend(task, url, model, orders=args.orders, timeout=args.timeout, name=name)
    if name == "llm":
        return LLMBackend(
            task,
            args.llm_url,
            args.llm_model,
            extra=json.loads(args.llm_extra),
            timeout=args.timeout,
            api_key=os.environ.get(args.llm_key_env),
        )
    raise SystemExit(f"unknown backend {name!r} (choose from rules, jev, llm, or a --jev-server / --answers name)")


def _short(row: dict) -> str:
    return f"{row['id']}: {row['text'][:60]!r}"


def cmd_check(args: argparse.Namespace) -> int:
    task = load_task(args.task)
    problems = validate(task)
    for p in problems:
        print(f"problem: {p}")
    if problems:
        return 1

    gold = task.gold
    counts = Counter(accepted_labels(row)[0] for row in gold)
    sources = Counter(row.get("source", "real") for row in gold)
    splits = Counter(row.get("split", "unsplit") for row in gold)
    two_way = [row for row in gold if len(accepted_labels(row)) > 1]
    with_context = sum(bool(context_turns(row)) for row in gold)

    print(f"{task.name}: {len(gold)} gold rows {dict(sources)}, splits {dict(splits)}, {with_context} with context")
    for label in task.labels:
        print(f"  {label:<20} {counts.get(label, 0)}")
    if gold and len(two_way) / len(gold) > MAX_TWO_WAY_SHARE:
        print(f"warning: {len(two_way)} two-way rows ({len(two_way) / len(gold):.0%}); keep it under {MAX_TWO_WAY_SHARE:.0%} or scores get soft")

    keys = [row_key(row) for row in gold]
    for i, j in near_copies(keys):
        print(f"warning: near-duplicate gold rows {_short(gold[i])} and {_short(gold[j])}")

    leaked = 0
    if args.against:
        others = read_jsonl(Path(args.against))
        for i, j in near_copies(keys, [row_key(o) for o in others]):
            print(f"LEAK: gold {_short(gold[i])} is in {args.against} as {others[j]['text'][:60]!r}")
            leaked += 1
        print(f"{leaked} gold rows leaked into {args.against}" if leaked else f"no gold rows found in {args.against}")

    real = sources.get("real", 0)
    if real < GOLD_TARGET:
        print(f"{GOLD_TARGET - real} more real examples to reach {GOLD_TARGET}. Label them with: python -m zsbench label {args.task} <inputs>")
    return 1 if leaked else 0


def cmd_run(args: argparse.Namespace) -> int:
    task = load_task(args.task)
    problems = validate(task)
    if problems:
        print("\n".join(f"problem: {p}" for p in problems), file=sys.stderr)
        return 1
    rows = [row for row in task.gold if not args.split or row.get("split") == args.split]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print(f"no gold rows in {task.gold_path}" + (f" with split {args.split!r}" if args.split else ""), file=sys.stderr)
        return 1

    out = Path(args.out or f"runs/{task.name}-{datetime.now():%Y%m%d-%H%M%S}")
    out.mkdir(parents=True, exist_ok=True)
    failed = False
    for name in args.backends.split(","):
        backend = make_backend(name.strip(), task, args)
        try:
            backend.check()
        except BackendError as e:
            print(f"{backend.name}: skipped, server not reachable ({e})", file=sys.stderr)
            failed = True
            continue

        preds = []
        for i, row in enumerate(rows, 1):
            accepted = accepted_labels(row)
            start = time.perf_counter()
            result = backend.predict(row["text"], context_turns(row))
            elapsed_ms = (time.perf_counter() - start) * 1000
            pred = {
                "id": row["id"],
                "split": row.get("split"),
                "text": row["text"],
                "gold": accepted[0],
                "accepted": accepted,
                "pred": result["pred"],
                "confidence": result.get("confidence"),
                "probs": result.get("probs"),
                "latency_ms": result.get("latency_ms", elapsed_ms),
            }
            if "extra" in result:
                pred["extra"] = result["extra"]
            preds.append(pred)
            print(f"\r{backend.name}: {i}/{len(rows)}", end="", file=sys.stderr)
        print(file=sys.stderr)

        (out / f"{backend.name}.jsonl").write_text(
            "".join(json.dumps(p) + "\n" for p in preds), encoding="utf-8"
        )
        meta = {
            "backend": backend.name,
            "task": task.name,
            "labels": list(task.labels),
            "split": args.split,
            "n": len(preds),
            "finished": datetime.now().isoformat(timespec="seconds"),
            **backend.describe(),
        }
        (out / f"{backend.name}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if any(out.glob("*.meta.json")):
        headline, full = render(out)
        (out / "report.md").write_text(full, encoding="utf-8")
        print(headline)
        print(f"\nfull report: {out / 'report.md'}")
    return 1 if failed else 0


def cmd_prompts(args: argparse.Namespace) -> int:
    task = load_task(args.task)
    rows = [row for row in task.gold if not args.split or row.get("split") == args.split]
    out = Path(args.out)
    if out.exists():
        print(f"{out} exists; it may hold collected replies. Pick another --out.", file=sys.stderr)
        return 1
    # No gold labels in this file: whoever collects the replies shouldn't see them.
    lines = [
        {
            "id": row["id"],
            "split": row.get("split"),
            "key": row_key(row),
            "prompt": chat_prompt(task, row["text"], context_turns(row)),
            "reply": None,
        }
        for row in rows
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    print(f"{len(lines)} prompts -> {out}")
    print(f'Fill in each "reply", then: python -m zsbench run {args.task} --backends NAME --answers NAME={out}')
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    headline, full = render(args.run_dir)
    (Path(args.run_dir) / "report.md").write_text(full, encoding="utf-8")
    print(headline)
    return 0


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def cmd_generate(args: argparse.Namespace) -> int:
    task = load_task(args.task)
    path = task.path / "candidates.jsonl"
    kinds = [k.strip() for k in args.kinds.split(",")] if args.kinds else None
    unknown = [k for k in kinds or [] if k not in KINDS]
    if unknown:
        print(f"unknown kinds {unknown}; choose from {', '.join(KINDS)}", file=sys.stderr)
        return 1
    gen = make_generator(task, args.model, args.url, os.environ.get(args.key_env),
                         temperature=args.temperature, effort=args.effort)
    try:
        gen.check()
    except (BackendError, GenerationError) as e:
        print(f"generator not reachable: {e}", file=sys.stderr)
        return 1
    log = lambda msg: print(msg, file=sys.stderr)  # noqa: E731
    rows = generate(task, gen, args.n, batch=args.batch, existing=read_jsonl(path), seed=args.seed,
                    workers=args.workers, kinds=kinds, log=log)
    _append_jsonl(path, rows)
    print(f"{len(rows)} new candidates -> {path}")
    print(f"  by intended label: {dict(Counter(r['intended'] for r in rows))}")
    print(f"  by kind: {dict(Counter(r['kind'] for r in rows))}")
    return 0 if rows else 1


def cmd_blindcheck(args: argparse.Namespace) -> int:
    task = load_task(args.task)
    teacher_path, hand_path = task.path / "teacher.jsonl", task.path / "to_label.jsonl"
    done = {r["id"] for r in read_jsonl(teacher_path) + read_jsonl(hand_path) + task.gold}
    todo = take_whole_groups([c for c in read_jsonl(task.path / "candidates.jsonl") if c["id"] not in done], args.limit)
    if not todo:
        print("nothing to check: run `generate` first, or every candidate is already sorted", file=sys.stderr)
        return 1

    key = os.environ.get(args.key_env)
    extra = json.loads(args.extra)

    def backend(model):
        return LLMBackend(task, args.url, model, extra=extra, timeout=args.timeout, api_key=key)

    checkers = [backend(m.strip()) for m in args.checkers.split(",") if m.strip()]
    tiebreaker = backend(args.tiebreaker) if args.tiebreaker else None
    try:
        checkers[0].check()
    except BackendError as e:
        print(f"checkers not reachable: {e}", file=sys.stderr)
        return 1

    log = lambda msg: print(msg, file=sys.stderr)  # noqa: E731
    teacher, hand, summary = blindcheck(todo, checkers, tiebreaker, workers=args.workers, log=log)
    _append_jsonl(teacher_path, teacher)
    _append_jsonl(hand_path, hand)
    print(f"{summary['n']} candidates checked: {summary['piles']}")
    for model, rate in summary["checker_agrees_with_generator"].items():
        print(f"  {model}: agrees with generator {rate:.0%}, invalid {summary['checker_invalid'][model]}")
    print(f"{len(teacher)} -> {teacher_path}")
    print(f"{len(hand)} -> {hand_path}  (label: python -m zsbench label {args.task} {hand_path} --source synthetic)")
    return 0


def read_inputs(path: Path) -> list[dict]:
    """.txt: one first message per line. .jsonl: {"text", "context"?} per line."""
    if path.suffix == ".jsonl":
        return [{"text": r["text"], **({"context": r["context"]} if r.get("context") else {})} for r in read_jsonl(path)]
    return [{"text": line} for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_answer(answer: str, labels: list[str]) -> list[str] | None:
    """'2' -> one label, '2,3' -> a two-way row; None if the answer isn't valid."""
    parts = [p.strip() for p in answer.split(",")]
    if not 1 <= len(parts) <= 2 or not all(p.isdigit() and 1 <= int(p) <= len(labels) for p in parts):
        return None
    picked = [labels[int(p) - 1] for p in parts]
    return picked if len(set(picked)) == len(picked) else None


def cmd_label(args: argparse.Namespace) -> int:
    """Hand-label raw agent inputs into gold.jsonl. Resumable: already-labeled inputs are skipped."""
    task = load_task(args.task)
    labels = list(task.labels)
    done = {row_key(row) for row in task.gold}
    todo = [r for r in read_inputs(Path(args.inputs)) if row_key(r) not in done]
    real = sum(row.get("source", "real") == "real" for row in task.gold)

    print(f"{len(todo)} unlabeled inputs, {real}/{GOLD_TARGET} real gold rows so far.")
    for n, label in enumerate(labels, 1):
        print(f"  {n}. {label}: {task.labels[label]}")
    print("  2,3 = genuinely two-way (use sparingly), u = undo last, s = skip, q = quit\n")

    written: list[tuple[str, int]] = []  # (row id, todo index) labeled this session, for undo
    i = 0
    while i < len(todo):
        item = todo[i]
        for turn in context_turns(item):
            print(f"  | {turn}")
        print(f"\n{item['text']}")
        while True:
            answer = input(f"[1-{len(labels)} / u / s / q] > ").strip().lower()
            picked = parse_answer(answer, labels)
            if answer in ("u", "s", "q") or picked:
                break
        if answer == "q":
            break
        if answer == "u":
            if written:
                row_id, i = written.pop()
                remove_rows(task.gold_path, {row_id})
                print("  undone, showing it again:\n")
            else:
                print("  nothing to undo this session\n")
            continue
        if picked:
            row = {
                "id": hashlib.sha1(row_key(item).encode("utf-8")).hexdigest()[:12],
                "text": item["text"],
                **({"context": item["context"]} if "context" in item else {}),
                "label": picked[0] if len(picked) == 1 else picked,
                "source": args.source,
            }
            _append_jsonl(task.gold_path, [row])
            written.append((row["id"], i))
        i += 1
    real = sum(row.get("source", "real") == "real" for row in read_jsonl(task.gold_path))
    print(f"\n{real}/{GOLD_TARGET} real gold rows.")
    return 0


def cmd_route(args: argparse.Namespace) -> int:
    """One routing decision as JSON; exit 1 when the router fell back."""
    decision = Router(fallback=args.fallback, timeout=args.timeout).route(args.text, args.context)
    print(json.dumps(decision.to_dict()))
    return 0 if decision.ok else 1


def cmd_split(args: argparse.Namespace) -> int:
    task = load_task(args.task)
    problems = validate(task)
    if problems:
        print("\n".join(f"problem: {p}" for p in problems), file=sys.stderr)
        return 1
    before = Counter(r.get("split", "untagged") for r in task.gold)
    counts = assign_splits(task, dev_frac=args.dev_frac, seed=args.seed)
    write_gold(task)
    print(f"{task.gold_path}: {dict(counts)} (was {dict(before)})")
    by_label = Counter((accepted_labels(r)[0], r["split"]) for r in task.gold)
    for label in task.labels:
        print(f"  {label:<20} dev {by_label[(label, 'dev')]:>3}   test {by_label[(label, 'test')]:>3}")
    return 0


def remove_rows(path: Path, ids: set[str]) -> None:
    rows = [r for r in read_jsonl(path) if r.get("id") not in ids]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def cmd_adjudicate(args: argparse.Namespace) -> int:
    task = load_task(args.task)
    hand_path, teacher_path = task.path / "to_label.jsonl", task.path / "teacher.jsonl"
    hand = read_jsonl(hand_path)
    if not hand:
        print(f"nothing to settle: {hand_path} is empty", file=sys.stderr)
        return 1
    judge = ClaudeAdjudicator(task, args.model, effort=args.effort)
    try:
        judge.check()
    except GenerationError as e:
        print(f"adjudicator not reachable: {e}", file=sys.stderr)
        return 1

    log = lambda msg: print(msg, file=sys.stderr)  # noqa: E731
    settled, still_open, done = adjudicate(task, hand, judge, workers=args.workers, log=log)
    archive = task.path / "archive"
    archive.mkdir(exist_ok=True)
    shutil.copy(hand_path, archive / f"to_label.{datetime.now():%Y%m%d-%H%M%S}.jsonl")  # keeps the votes
    _append_jsonl(teacher_path, settled)
    hand_path.write_text("".join(json.dumps(r) + "\n" for r in still_open), encoding="utf-8")
    overruled = sum(r["label"] != r["intended"] for r in settled)
    print(f"{len(done)} already in gold (removed from the queue)")
    print(f"{len(settled)} settled by {args.model} -> {teacher_path} (overruled the generator on {overruled})")
    print(f"{len(still_open)} still need you in {hand_path} (pair partners of rows you labeled, or failed calls)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="zsbench", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check", help="validate a task, count labels, flag near-duplicates and leaks")
    p.add_argument("task")
    p.add_argument("--against", help="training .jsonl to check for gold leakage (exit 1 if any)")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("route", help="ask the production router (DeepSeek V4.1 Flash on DeepInfra) for the next action")
    p.add_argument("text", help="the latest user message")
    p.add_argument("--context", action="append", default=[], help="an earlier turn, oldest first (repeatable)")
    p.add_argument("--fallback", default="answer_directly", help="label to return if the call fails")
    p.add_argument("--timeout", type=float, default=15)
    p.set_defaults(func=cmd_route)

    p = sub.add_parser("split", help="tag untagged gold rows dev/test; pairs and near-copies stay together, "
                                     "existing tags never move")
    p.add_argument("task")
    p.add_argument("--dev-frac", type=float, default=1 / 3, help="share of each label kept in dev")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("prompts", help="write one self-contained chat prompt per gold row (full label menu "
                                       "each time, no gold labels) for scoring a chat app")
    p.add_argument("task")
    p.add_argument("--split", choices=SPLITS, help="only rows with this split (default: all rows)")
    p.add_argument("--out", required=True, help="new .jsonl; fill in each reply, then pass it to run --answers")
    p.set_defaults(func=cmd_prompts)

    p = sub.add_parser("run", help="run backends over the gold set and write a report")
    p.add_argument("task")
    p.add_argument("--backends", default="rules,jev,llm",
                   help="comma-separated: rules, jev, llm, and any --jev-server or --answers names")
    p.add_argument("--jev-server", action="append", default=[], metavar="NAME=URL[,MODEL]",
                   help="another System One server to score as backend NAME (repeatable), "
                        "e.g. kev=http://127.0.0.1:8009,kev-latest")
    p.add_argument("--answers", action="append", default=[], metavar="NAME=FILE[,MODEL]",
                   help="score collected replies (a filled-in `prompts` file) as backend NAME (repeatable), "
                        "e.g. snapchat=runs/snap/dev.jsonl,snapchat-my-ai")
    p.add_argument("--split", choices=SPLITS, help="only rows with this split (default: all rows)")
    p.add_argument("--out", help="run directory; reuse one to add backends run at different times")
    p.add_argument("--limit", type=int, default=0, help="only the first N gold rows")
    p.add_argument("--timeout", type=float, default=60)
    p.add_argument("--jev-url", default="http://127.0.0.1:8000")
    p.add_argument("--jev-model", default=DEFAULT_MODEL)
    p.add_argument("--orders", type=int, default=0, help="label orders per Jev call; 0 = one per label, 1 = no debiasing")
    p.add_argument("--llm-url", default=os.environ.get("LLM_URL", "http://127.0.0.1:8001/v1"))
    p.add_argument("--llm-model", default=os.environ.get("LLM_MODEL", DEFAULT_MODEL))
    p.add_argument(
        "--llm-key-env",
        default="LLM_API_KEY",
        help="env var holding the API key for --llm-url, sent as a Bearer token; unset means no auth (local vLLM)",
    )
    p.add_argument(
        "--llm-extra",
        default=os.environ.get("LLM_EXTRA", '{"chat_template_kwargs": {"enable_thinking": false}}'),
        help="JSON merged into each chat request; the default turns off Qwen thinking on vLLM",
    )
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("report", help="rebuild report.md for a run directory")
    p.add_argument("run_dir")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("label", help="hand-label raw inputs (.txt one per line, or .jsonl with text/context) into gold.jsonl")
    p.add_argument("task")
    p.add_argument("inputs")
    p.add_argument("--source", default="real", help="source tag for new rows: real (agent traffic) or synthetic")
    p.set_defaults(func=cmd_label)

    def add_remote_args(p, model_env, model_default, model_help):
        p.add_argument("task")
        p.add_argument("--url", default=os.environ.get("LLM_URL", "http://127.0.0.1:8001/v1"),
                       help="OpenAI-compatible base URL (default: LLM_URL from .env)")
        p.add_argument("--key-env", default="LLM_API_KEY", help="env var holding the API key")
        p.add_argument("--timeout", type=float, default=300)
        p.add_argument(model_env[1], default=os.environ.get(model_env[0], model_default), help=model_help)

    p = sub.add_parser("generate", help="write hard candidate inputs with an intended label into candidates.jsonl")
    add_remote_args(p, ("GEN_MODEL", "--model"), DEFAULT_GENERATOR,
                    "generator model; claude-* models go through the Claude API (ANTHROPIC_API_KEY)")
    p.add_argument("--n", type=int, default=100, help="about how many new candidates to add")
    p.add_argument("--kinds", help=f"comma-separated subset of: {', '.join(KINDS)} (default: all)")
    p.add_argument("--batch", type=int, default=8, help="candidates per generator call")
    p.add_argument("--temperature", type=float, default=0.9, help="OpenAI-compatible generators only")
    p.add_argument("--effort", choices=["low", "medium", "high"],
                   help="Claude generators only (default: low for Haiku, medium otherwise)")
    p.add_argument("--seed", type=int, help="fixes the kind/topic sequence (default: random)")
    p.add_argument("--workers", type=int, default=4, help="parallel generator calls")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("blindcheck", help="checkers label candidates blind; agreed -> teacher.jsonl, disputed -> to_label.jsonl")
    add_remote_args(p, ("CHECK_MODELS", "--checkers"), DEFAULT_CHECKERS, "comma-separated checker models")
    p.add_argument("--tiebreaker", default=os.environ.get("TIEBREAK_MODEL", DEFAULT_TIEBREAKER),
                   help="model that votes when the checkers split; empty string to disable")
    p.add_argument("--extra", default=os.environ.get("CHECK_EXTRA", "{}"), help="JSON merged into each checker request")
    p.add_argument("--limit", type=int, default=0, help="check only about N candidates (whole pairs are kept together)")
    p.add_argument("--workers", type=int, default=8, help="candidates checked in parallel")
    p.set_defaults(func=cmd_blindcheck)

    p = sub.add_parser("adjudicate", help="a Claude model settles disputed rows you haven't labeled; "
                                          "they go to teacher.jsonl, never gold")
    p.add_argument("task")
    p.add_argument("--model", default=os.environ.get("ADJUDICATE_MODEL", "claude-sonnet-5-5"))
    p.add_argument("--effort", choices=["low", "medium", "high"], default="medium")
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(func=cmd_adjudicate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    load_dotenv(ENV_FILE)
    sys.exit(main())
