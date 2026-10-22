import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.fakes import FakeServer, jev_respond, llm_respond
from zsbench.__main__ import load_dotenv, main
from zsbench.task import load_task

TASK_DIR = Path(__file__).resolve().parent.parent / "tasks" / "tool_router"


def quiet(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class CliTest(unittest.TestCase):
    def test_run_all_backends_then_report(self):
        with tempfile.TemporaryDirectory() as tmp, FakeServer(jev_respond) as jev, FakeServer(llm_respond) as llm:
            code, out, _ = quiet([
                "run", str(TASK_DIR), "--out", tmp,
                "--jev-url", jev.url, "--llm-url", llm.url + "/v1",
            ])
            self.assertEqual(code, 0)
            for backend in ("rules", "jev", "llm"):
                self.assertIn(f"| {backend} |", out)
                rows = (Path(tmp) / f"{backend}.jsonl").read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(rows), len(load_task(TASK_DIR).gold))  # the live gold set grows
            report = (Path(tmp) / "report.md").read_text(encoding="utf-8")
            self.assertIn("## Confusion: jev", report)

            code, out, _ = quiet(["report", tmp])
            self.assertEqual(code, 0)
            self.assertIn("| llm |", out)

    def test_llm_key_read_from_named_env_var(self):
        with tempfile.TemporaryDirectory() as tmp, FakeServer(llm_respond) as llm, \
                mock.patch.dict("os.environ", {"DEEPINFRA_API_KEY": "sk-env"}):
            code, _, _ = quiet([
                "run", str(TASK_DIR), "--out", tmp, "--backends", "llm", "--limit", "2",
                "--llm-url", llm.url + "/v1", "--llm-key-env", "DEEPINFRA_API_KEY",
            ])
            self.assertEqual(code, 0)
            self.assertEqual({r["headers"]["Authorization"] for r in llm.requests}, {"Bearer sk-env"})
            self.assertNotIn("sk-env", (Path(tmp) / "llm.meta.json").read_text(encoding="utf-8"))

    def test_dotenv_fills_llm_defaults(self):
        with tempfile.TemporaryDirectory() as tmp, FakeServer(llm_respond) as llm, \
                mock.patch.dict("os.environ", {"LLM_MODEL": "already-set"}, clear=True):
            env = Path(tmp) / ".env"
            env.write_text(
                "﻿# comment\n"
                "LLM_API_KEY= 'sk-file' \n"
                f"LLM_URL={llm.url}/v1\n"
                "LLM_MODEL=from-file\n"
                "LLM_EXTRA={}\n"
                "EMPTY=\n",
                encoding="utf-8",
            )
            load_dotenv(env)
            load_dotenv(Path(tmp) / "missing.env")  # no-op
            self.assertNotIn("EMPTY", os.environ)

            code, _, _ = quiet(["run", str(TASK_DIR), "--out", str(Path(tmp) / "run"), "--backends", "llm", "--limit", "1"])
            self.assertEqual(code, 0)
            body, headers = llm.requests[0]["body"], llm.requests[0]["headers"]
            self.assertEqual(headers["Authorization"], "Bearer sk-file")
            self.assertEqual(body["model"], "already-set")  # a real env var beats .env
            self.assertNotIn("chat_template_kwargs", body)

    def test_named_jev_servers_become_separate_backends(self):
        with tempfile.TemporaryDirectory() as tmp, FakeServer(jev_respond) as kev, FakeServer(jev_respond) as intern:
            code, out, _ = quiet([
                "run", str(TASK_DIR), "--out", tmp, "--backends", "kev,intern", "--limit", "3",
                "--jev-server", f"kev={kev.url},kev-latest", "--jev-server", f"intern={intern.url}",
            ])
            self.assertEqual(code, 0)
            self.assertIn("| kev | kev-latest |", out)
            self.assertIn("| intern | intern |", out)
            self.assertEqual({r["body"]["model"] for r in kev.requests}, {"kev-latest"})
            self.assertEqual(len(intern.requests), 3)

    def test_bad_jev_server_spec_is_rejected(self):
        with self.assertRaises(SystemExit):
            quiet(["run", str(TASK_DIR), "--backends", "kev", "--jev-server", "kev"])

    def test_unreachable_backend_is_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out, err = quiet([
                "run", str(TASK_DIR), "--out", tmp, "--backends", "rules,jev",
                "--jev-url", "http://127.0.0.1:9", "--limit", "5",
            ])
            self.assertEqual(code, 1)
            self.assertIn("jev: skipped", err)
            self.assertIn("| rules |", out)
            self.assertFalse((Path(tmp) / "jev.jsonl").exists())

    def test_backends_run_separately_merge_into_one_report(self):
        with tempfile.TemporaryDirectory() as tmp, FakeServer(llm_respond) as llm:
            quiet(["run", str(TASK_DIR), "--out", tmp, "--backends", "rules"])
            code, out, _ = quiet(["run", str(TASK_DIR), "--out", tmp, "--backends", "llm", "--llm-url", llm.url + "/v1"])
            self.assertEqual(code, 0)
            self.assertIn("| rules |", out)
            self.assertIn("| llm |", out)

    def test_label_with_context_and_two_way_then_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp) / "t"
            task_dir.mkdir()
            (task_dir / "task.json").write_text(json.dumps({
                "name": "t", "question": "q?", "labels": {"yes": "y", "no": "n", "maybe": "m"},
            }), encoding="utf-8")
            inputs = Path(tmp) / "inputs.jsonl"
            inputs.write_text("".join(json.dumps(r) + "\n" for r in [
                {"text": "first input"},
                {"text": "second input"},
                {"text": "do it", "context": ["assistant: shall I?"]},
                {"text": "do it"},
            ]), encoding="utf-8")

            # "x" and "1,1" are rejected and re-asked.
            answers = ["2", "s", "x", "1,1", "1,3", "q"]
            with mock.patch("builtins.input", side_effect=answers):
                code, _, _ = quiet(["label", str(task_dir), str(inputs)])
            self.assertEqual(code, 0)
            gold = [json.loads(line) for line in (task_dir / "gold.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(
                [(g["text"], g.get("context"), g["label"]) for g in gold],
                [("first input", None, "no"), ("do it", ["assistant: shall I?"], ["yes", "maybe"])],
            )

            # Resuming skips labeled inputs; the same text without context is a different input.
            with mock.patch("builtins.input", side_effect=["s", "1"]):
                quiet(["label", str(task_dir), str(inputs)])
            gold = [json.loads(line) for line in (task_dir / "gold.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(gold), 3)
            self.assertEqual((gold[2]["text"], gold[2].get("context")), ("do it", None))

            code, out, _ = quiet(["check", str(task_dir)])
            self.assertEqual(code, 0)
            self.assertIn("197 more real examples", out)
            self.assertIn("1 with context", out)
            self.assertIn("warning: 1 two-way rows (33%)", out)

    def test_label_undo_reshows_the_last_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp) / "t"
            task_dir.mkdir()
            (task_dir / "task.json").write_text(json.dumps({
                "name": "t", "question": "q?", "labels": {"yes": "y", "no": "n"},
            }), encoding="utf-8")
            inputs = Path(tmp) / "inputs.txt"
            inputs.write_text("first\nsecond\n", encoding="utf-8")
            # Undo with nothing labeled is a no-op; then label "no", undo it, relabel "yes", quit.
            with mock.patch("builtins.input", side_effect=["u", "2", "u", "1", "q"]):
                code, out, _ = quiet(["label", str(task_dir), str(inputs)])
            self.assertEqual(code, 0)
            self.assertIn("nothing to undo", out)
            gold = [json.loads(line) for line in (task_dir / "gold.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([(g["text"], g["label"]) for g in gold], [("first", "yes")])

    def make_split_task(self, tmp):
        task_dir = Path(tmp) / "t"
        task_dir.mkdir()
        (task_dir / "task.json").write_text(json.dumps({
            "name": "t", "question": "q?", "labels": {"yes": "y", "no": "n"},
            "rules": [{"label": "yes", "patterns": ["yes"]}], "default_label": "no",
        }), encoding="utf-8")
        gold = [
            {"id": "p1", "text": "pair half one about invoices", "label": "yes"},
            {"id": "p2", "text": "pair half two about refunds", "label": "no"},
            {"id": "n1", "text": "Run the full test suite and report failures", "label": "yes"},
            {"id": "n2", "text": "run the full test suite and report the failures!", "label": "yes"},
            {"id": "old", "text": "already tagged", "label": "no", "split": "test"},
        ] + [{"id": f"s{i}", "text": f"single row number {i} {'yes' if i % 2 else 'nope'}",
              "label": "yes" if i % 2 else "no"} for i in range(12)]
        (task_dir / "gold.jsonl").write_text("".join(json.dumps(r) + "\n" for r in gold), encoding="utf-8")
        (task_dir / "candidates.jsonl").write_text("".join(json.dumps({"id": i, "group": "g-pair"}) + "\n"
                                                           for i in ("p1", "p2")), encoding="utf-8")
        return task_dir

    def test_split_keeps_groups_together_and_never_moves_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = self.make_split_task(tmp)
            code, out, _ = quiet(["split", str(task_dir)])
            self.assertEqual(code, 0)
            split = {r["id"]: r["split"] for r in load_task(task_dir).gold}
            self.assertEqual(split["p1"], split["p2"])  # contrastive pair (via candidates.jsonl)
            self.assertEqual(split["n1"], split["n2"])  # near-copies
            self.assertEqual(split["old"], "test")
            self.assertTrue(4 <= list(split.values()).count("dev") <= 8)  # about a third of 17

            before = (task_dir / "gold.jsonl").read_text(encoding="utf-8")
            quiet(["split", str(task_dir), "--seed", "99"])  # every row tagged: nothing moves
            self.assertEqual((task_dir / "gold.jsonl").read_text(encoding="utf-8"), before)

    def test_test_only_run_hides_confusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = self.make_split_task(tmp)
            quiet(["split", str(task_dir)])
            quiet(["run", str(task_dir), "--backends", "rules", "--split", "test", "--out", str(Path(tmp) / "run")])
            report = (Path(tmp) / "run" / "report.md").read_text(encoding="utf-8")
            self.assertNotIn("## Confusion", report)
            self.assertIn("hidden", report)

    def test_report_headlines_test_and_lists_only_dev_misses(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = self.make_split_task(tmp)
            quiet(["split", str(task_dir)])
            gold = load_task(task_dir).gold
            code, out, _ = quiet(["run", str(task_dir), "--backends", "rules", "--out", str(Path(tmp) / "run")])
            self.assertEqual(code, 0)
            self.assertIn("Scope: test split", out)
            n_test = sum(r["split"] == "test" for r in gold)
            self.assertIn(f"| rules | - | {n_test} |", out)
            report = (Path(tmp) / "run" / "report.md").read_text(encoding="utf-8")
            self.assertIn("## Dev misses", report)
            self.assertIn("## By split", report)
            # rules say "no" unless the text contains "yes": every "yes" row without it is a miss.
            missed = [r for r in gold if r["label"] == "yes" and "yes" not in r["text"]]
            for r in missed:
                self.assertEqual(r["text"] in report, r["split"] == "dev")

    def test_check_flags_near_duplicates_and_leaks(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp) / "t"
            task_dir.mkdir()
            (task_dir / "task.json").write_text(json.dumps({
                "name": "t", "question": "q?", "labels": {"yes": "y", "no": "n"},
            }), encoding="utf-8")
            (task_dir / "gold.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [
                {"id": "1", "text": "Run the full test suite and report failures", "label": "yes"},
                {"id": "2", "text": "run the full test suite and report the failures!", "label": "yes"},
                {"id": "3", "text": "What is a tuple", "label": "no"},
            ]), encoding="utf-8")
            train = Path(tmp) / "train.jsonl"
            train.write_text(json.dumps({"text": "what is a TUPLE?", "label": "no"}) + "\n", encoding="utf-8")

            code, out, _ = quiet(["check", str(task_dir)])
            self.assertEqual(code, 0)
            self.assertIn("near-duplicate gold rows 1:", out)

            code, out, _ = quiet(["check", str(task_dir), "--against", str(train)])
            self.assertEqual(code, 1)
            self.assertIn("LEAK: gold 3:", out)
            self.assertIn("1 gold rows leaked", out)

    def test_run_split_filters_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp) / "t"
            task_dir.mkdir()
            (task_dir / "task.json").write_text(json.dumps({
                "name": "t", "question": "q?", "labels": {"yes": "y", "no": "n"},
            }), encoding="utf-8")
            (task_dir / "gold.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [
                {"id": "1", "text": "a", "label": "yes", "split": "dev"},
                {"id": "2", "text": "b", "label": "no", "split": "test"},
                {"id": "3", "text": "c", "label": "yes", "split": "test"},
            ]), encoding="utf-8")
            code, out, _ = quiet(["run", str(task_dir), "--backends", "rules", "--split", "test", "--out", str(Path(tmp) / "run")])
            self.assertEqual(code, 0)
            self.assertIn("| rules | - | 2 |", out)
            self.assertIn("split: test", (Path(tmp) / "run" / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
