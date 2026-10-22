import contextlib
import io
import json
import shutil
import tempfile
import unittest
import unittest.mock
from collections import Counter
from pathlib import Path

from types import SimpleNamespace

from tests.fakes import FakeServer
from zsbench.__main__ import main
from zsbench.backends import LLMBackend
from zsbench.synth import (
    KINDS,
    ClaudeAdjudicator,
    ClaudeGenerator,
    adjudicate,
    Generator,
    blindcheck,
    generate,
    make_generator,
    take_whole_groups,
    vote,
)
from zsbench.task import load_task, read_jsonl

TASK_DIR = Path(__file__).resolve().parent.parent / "tasks" / "tool_router"

GENERATED = [
    # Same text as gold seed-22: dropped as a near-copy.
    {"text": "Check the changelog for what changed in 0.9.", "context": [], "intended": "read_file", "pair_id": ""},
    {"text": "is the vllm 0.9 changelog run_code up online", "context": [], "intended": "web_search", "pair_id": "p1"},
    {"text": "is the vllm 0.9 changelog in our docs folder", "context": [], "intended": "read_file", "pair_id": "p1"},
    {"text": "ok go", "context": ["assistant: Should I run the benchmark now?"], "intended": "run_code", "pair_id": ""},
    {"text": "  ", "context": [], "intended": "run_code", "pair_id": ""},
    {"text": "something", "context": [], "intended": "not_a_label", "pair_id": ""},
]


def respond(path, payload):
    """Generator calls get GENERATED. Checkers answer by model name:
    agree -> the intended label, read from a hint table; contrarian -> always ask_user; broken -> not JSON."""
    fmt = payload["response_format"]["json_schema"]["name"]
    if fmt == "candidates":
        content = json.dumps({"examples": GENERATED})
    else:
        text = payload["messages"][-1]["content"]
        hints = {"online": "web_search", "docs folder": "read_file", "benchmark": "run_code"}
        label = next((v for k, v in hints.items() if k in text), "answer_directly")
        model = payload["model"]
        if model == "contrarian":
            label = "ask_user"
        content = "nope" if model == "broken" else json.dumps({"label": label})
    return 200, {"choices": [{"message": {"role": "assistant", "content": content}}]}


def candidate(cid, text, intended, group=None):
    return {"id": cid, "text": text, "intended": intended, "group": group or cid, "kind": "messy"}


class VoteTest(unittest.TestCase):
    def test_vote_table(self):
        never = lambda: self.fail("tiebreaker should not be asked")  # noqa: E731
        self.assertEqual(vote("a", ["a", "a"], never), ("teacher", "unanimous", None))
        self.assertEqual(vote("a", ["a", "b"], lambda: "a"), ("teacher", "tiebreak", "a"))
        self.assertEqual(vote("a", ["a", "b"], lambda: "b"), ("hand", None, "b"))
        self.assertEqual(vote("a", ["a", None], lambda: None), ("hand", None, None))
        self.assertEqual(vote("a", ["b", "b"], never), ("hand", None, None))  # checkers agree against the generator
        self.assertEqual(vote("a", ["a", "b"], None), ("hand", None, None))  # split with no tiebreaker
        self.assertEqual(vote("a", ["a", "a", "b"], never), ("teacher", "majority", None))

    def test_take_whole_groups_never_splits_a_pair(self):
        rows = [candidate("1", "x", "a", "g1"), candidate("2", "y", "a", "g2"), candidate("3", "z", "a", "g2"),
                candidate("4", "w", "a", "g3")]
        self.assertEqual([r["id"] for r in take_whole_groups(rows, 2)], ["1", "2", "3"])
        self.assertEqual(len(take_whole_groups(rows, 0)), 4)


class GenerateTest(unittest.TestCase):
    def test_generate_validates_dedups_and_groups(self):
        task = load_task(TASK_DIR)
        with FakeServer(respond) as server:
            gen = Generator(task, server.url + "/v1", "gen", api_key="sk-x")
            rows = generate(task, gen, n=10, batch=8, seed=1, workers=2, log=lambda m: None)

        self.assertEqual(sorted(r["text"] for r in rows), [
            "is the vllm 0.9 changelog in our docs folder",
            "is the vllm 0.9 changelog run_code up online",
            "ok go",
        ])
        pair = [r for r in rows if "changelog" in r["text"]]
        self.assertEqual(len({r["group"] for r in pair}), 1)
        ok_go = next(r for r in rows if r["text"] == "ok go")
        self.assertEqual(ok_go["context"], ["assistant: Should I run the benchmark now?"])
        self.assertEqual(ok_go["group"], ok_go["id"])

        body = server.requests[0]["body"]
        self.assertEqual(body["temperature"], 0.9)
        self.assertEqual(body["response_format"]["json_schema"]["schema"]["properties"]["examples"]["items"]
                         ["properties"]["intended"]["enum"], list(task.labels))
        self.assertEqual(server.requests[0]["headers"]["Authorization"], "Bearer sk-x")
        self.assertEqual(len(server.requests), 6)  # stops at the call cap when every batch is a duplicate


class FakeClaude:
    """Stands in for anthropic.Anthropic: records calls and returns canned Message-like objects."""

    def __init__(self, stop_reason="end_turn", reply=None):
        self.calls = []
        self.stop_reason = stop_reason
        self.reply = reply or (lambda kwargs: json.dumps({"examples": GENERATED}))
        self.messages = SimpleNamespace(create=lambda **kw: self._create("messages", kw))
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: self._create("beta", kw)))
        self.models = SimpleNamespace(retrieve=lambda model: SimpleNamespace(id=model))

    def _create(self, api, kwargs):
        self.calls.append((api, kwargs))
        text = self.reply(kwargs)
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        )


class ClaudeGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.task = load_task(TASK_DIR)

    def test_haiku_uses_plain_messages_with_structured_output(self):
        client = FakeClaude()
        gen = ClaudeGenerator(self.task, "claude-haiku-5-5", client=client)
        gen.check()
        rows = gen.batch("clear", "travel planning", 8, ["web_search"], "b0", "a busy parent")
        api, kw = client.calls[0]
        self.assertEqual(api, "messages")
        self.assertEqual(kw["output_config"]["effort"], "low")
        self.assertEqual(kw["output_config"]["format"]["schema"]["properties"]["examples"]["items"]
                         ["properties"]["intended"]["enum"], list(self.task.labels))
        self.assertNotIn("temperature", kw)
        self.assertNotIn("fallbacks", kw)
        self.assertIn("a busy parent", kw["messages"][0]["content"])
        self.assertEqual({r["generator"] for r in rows}, {"claude-haiku-5-5"})
        self.assertEqual({r["style"] for r in rows}, {"a busy parent"})

    def test_sonnet_opts_into_server_side_fallback(self):
        client = FakeClaude()
        ClaudeGenerator(self.task, "claude-sonnet-5-5", effort="medium", client=client).batch(
            "near_miss", "home lab servers", 8, ["run_code"], "b0", "a student")
        api, kw = client.calls[0]
        self.assertEqual(api, "beta")
        self.assertEqual(kw["fallbacks"], "default")
        self.assertEqual(kw["betas"], ["server-side-fallback-2026-07-01"])
        self.assertEqual(kw["output_config"]["effort"], "medium")

    def test_refusal_is_skipped_not_fatal(self):
        gen = ClaudeGenerator(self.task, "claude-haiku-5-5", client=FakeClaude(stop_reason="refusal"))
        logs = []
        self.assertEqual(generate(self.task, gen, n=4, seed=0, workers=1, log=logs.append), [])
        self.assertTrue(any("stopped: refusal" in m for m in logs))

    def test_make_generator_routes_by_model_name(self):
        self.assertIsInstance(make_generator(self.task, "deepseek-ai/x", "http://x", None), Generator)
        with unittest.mock.patch("anthropic.Anthropic", return_value=FakeClaude()):
            haiku = make_generator(self.task, "claude-haiku-5-5", "http://x", None)
            sonnet = make_generator(self.task, "claude-sonnet-5-5", "http://x", None)
        self.assertIsInstance(haiku, ClaudeGenerator)
        self.assertEqual((haiku.effort, sonnet.effort), ("low", "medium"))

    def test_boundary_batches_name_the_pair_and_keep_rival_out_of_checks(self):
        client = FakeClaude()
        gen = ClaudeGenerator(self.task, "claude-sonnet-5-5", client=client)
        generate(self.task, gen, n=40, seed=2, workers=2, kinds=["boundary"], log=lambda m: None)
        prompts = [kw["messages"][0]["content"] for _, kw in client.calls]
        pairs = {p.split("the line between ")[1].split(":")[0] for p in prompts}
        self.assertGreater(len(pairs), 1)  # rotates through boundary pairs
        self.assertIn("No giveaway keywords", prompts[0])
        props = client.calls[0][1]["output_config"]["format"]["schema"]["properties"]["examples"]["items"]["properties"]
        self.assertIn("rival", props)
        self.assertIn("why", props)

        rows = [candidate("b1", "sounds like a lookup", "web_search")]
        rows[0] |= {"rival": "answer_directly", "why": "salary data changes"}
        with FakeServer(respond) as server:
            teacher, hand, _ = blindcheck(rows, [LLMBackend(self.task, server.url + "/v1", "agree")], log=lambda m: None)
        self.assertNotIn("salary data changes", json.dumps(server.requests))
        self.assertEqual((teacher + hand)[0]["rival"], "answer_directly")

    def test_clear_kind_has_no_hardness_push(self):
        client = FakeClaude()
        ClaudeGenerator(self.task, "claude-haiku-5-5", client=client).batch("clear", "travel", 8, ["web_search"], "b", "s")
        prompt = client.calls[0][1]["messages"][0]["content"]
        self.assertNotIn("No giveaway keywords", prompt)
        self.assertNotIn("rival", client.calls[0][1]["output_config"]["format"]["schema"]["properties"]["examples"]
                         ["items"]["properties"])

    def test_kinds_filter(self):
        client = FakeClaude()
        gen = ClaudeGenerator(self.task, "claude-sonnet-5-5", client=client)
        generate(self.task, gen, n=50, seed=0, workers=2, kinds=["near_miss", "multi_step"], log=lambda m: None)
        used = {kw["messages"][0]["content"].split("this kind: ")[1][:20] for _, kw in client.calls}
        expected = {KINDS[k][:20] for k in ("near_miss", "multi_step")}
        self.assertTrue(used <= expected)


def first_option(kwargs):
    """Adjudicator stand-in: always picks the first allowed label."""
    options = kwargs["output_config"]["format"]["schema"]["properties"]["label"]["enum"]
    return json.dumps({"label": options[0], "why": "first option"})


class AdjudicateTest(unittest.TestCase):
    def setUp(self):
        self.task = load_task(TASK_DIR)
        self.task.gold.append({"id": "g1", "text": "labeled by hand", "label": "read_file", "source": "synthetic"})
        votes = {"gpt": "answer_directly", "mimo": "run_code"}
        self.hand = [
            {**candidate("g1", "labeled by hand", "read_file", "pair"), "votes": votes},
            {**candidate("x2", "its pair partner", "run_code", "pair"), "votes": votes},
            {**candidate("x3", "net 45 from june 12, due when", "run_code"), "votes": votes, "context": ["user: hi"]},
        ]

    def test_splits_done_open_and_settled(self):
        client = FakeClaude(reply=first_option)
        judge = ClaudeAdjudicator(self.task, "claude-sonnet-5-5", effort="medium", client=client)
        settled, still_open, done = adjudicate(self.task, self.hand, judge, log=lambda m: None)

        self.assertEqual([r["id"] for r in done], ["g1"])
        self.assertEqual([r["id"] for r in still_open], ["x2"])  # its pair partner is gold
        self.assertEqual(len(client.calls), 1)
        row = settled[0]
        self.assertEqual((row["id"], row["label"], row["agreement"]), ("x3", "answer_directly", "adjudicated"))
        self.assertEqual(row["intended"], "run_code")
        self.assertEqual(row["context"], ["user: hi"])

        kw = client.calls[0][1]
        self.assertEqual(kw["output_config"]["format"]["schema"]["properties"]["label"]["enum"],
                         ["answer_directly", "run_code"])  # only the proposed labels, in task order
        self.assertEqual(kw["fallbacks"], "default")
        self.assertIn("Earlier turns:\n  user: hi", kw["messages"][0]["content"])

    def test_cli_moves_rows_and_archives_the_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp) / "t"
            task_dir.mkdir()
            shutil.copy(TASK_DIR / "task.json", task_dir / "task.json")
            (task_dir / "gold.jsonl").write_text(json.dumps(self.task.gold[-1]) + "\n", encoding="utf-8")
            (task_dir / "to_label.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.hand), encoding="utf-8")
            with unittest.mock.patch("anthropic.Anthropic", return_value=FakeClaude(reply=first_option)), \
                    contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["adjudicate", str(task_dir)]), 0)
            self.assertEqual([r["id"] for r in read_jsonl(task_dir / "teacher.jsonl")], ["x3"])
            self.assertEqual([r["id"] for r in read_jsonl(task_dir / "to_label.jsonl")], ["x2"])
            self.assertEqual(len(list((task_dir / "archive").glob("to_label.*.jsonl"))), 1)
            self.assertIn("overruled the generator on 1", out.getvalue())


class BlindcheckTest(unittest.TestCase):
    def setUp(self):
        self.task = load_task(TASK_DIR)
        self.candidates = [
            candidate("c1", "please check online", "web_search"),
            candidate("c2", "look in the docs folder", "read_file", "pair"),
            candidate("c3", "say hi", "ask_user", "pair"),  # contrarian matches, agree doesn't
            candidate("c4", "kick off the benchmark", "run_code"),
        ]

    def run_check(self, checkers, tiebreaker):
        with FakeServer(respond) as server:
            url = server.url + "/v1"
            backends = [LLMBackend(self.task, url, m) for m in checkers]
            tb = LLMBackend(self.task, url, tiebreaker) if tiebreaker else None
            result = blindcheck(self.candidates, backends, tb, workers=4, log=lambda m: None)
        return (*result, Counter(r["body"]["model"] for r in server.requests))

    def test_checkers_never_see_the_intended_label(self):
        with FakeServer(respond) as server:
            blindcheck(self.candidates, [LLMBackend(self.task, server.url + "/v1", "agree")], log=lambda m: None)
        for r in server.requests:
            self.assertNotIn("intended", json.dumps(r["body"]))

    def test_tiebreaker_resolves_splits_and_pairs_stay_together(self):
        teacher, hand, summary, calls = self.run_check(["agree", "contrarian"], "agree")
        # c1, c4: split agree/contrarian, tiebreaker sides with intended -> teacher.
        # c2: split, tiebreak agrees -> teacher, but its pair partner c3 is disputed -> pulled to hand.
        # c3: agree says answer_directly, contrarian says ask_user (= intended) -> split -> tiebreak says answer_directly -> hand.
        self.assertEqual(sorted(r["id"] for r in teacher), ["c1", "c4"])
        self.assertEqual(sorted(r["id"] for r in hand), ["c2", "c3"])
        self.assertEqual({r["agreement"] for r in teacher}, {"tiebreak"})
        self.assertEqual(summary["piles"], {"tiebreak": 2, "pulled_by_pair": 1, "disputed": 1})
        self.assertEqual(calls["agree"], 4 + 4)  # as checker on all 4, as tiebreaker on the 4 splits
        self.assertEqual(teacher[0]["source"], "teacher")
        self.assertIn("label", teacher[0])
        self.assertNotIn("label", hand[0])
        self.assertIn("tiebreak:agree", hand[0]["votes"])

    def test_unanimous_rows_skip_the_tiebreaker(self):
        teacher, hand, summary, calls = self.run_check(["agree", "agree2"], "contrarian")
        # No splits: c1, c2, c4 are unanimous, and on c3 both checkers agree against the generator.
        self.assertEqual(calls["contrarian"], 0)
        self.assertEqual(sorted(r["id"] for r in teacher), ["c1", "c4"])
        self.assertEqual(sorted(r["id"] for r in hand), ["c2", "c3"])  # c2 follows its disputed pair partner
        self.assertEqual(summary["checker_agrees_with_generator"]["agree"], 0.75)

    def test_duplicate_checkers_are_rejected(self):
        with self.assertRaises(ValueError):
            blindcheck(self.candidates, [LLMBackend(self.task, "http://x", "m"), LLMBackend(self.task, "http://x", "m")])

    def test_invalid_votes_go_to_hand(self):
        teacher, hand, summary, _ = self.run_check(["agree", "broken"], None)
        self.assertEqual(teacher, [])
        self.assertEqual(len(hand), 4)
        self.assertEqual(summary["checker_invalid"], {"agree": 0, "broken": 4})


class SynthCliTest(unittest.TestCase):
    def test_generate_then_blindcheck_then_resume(self):
        with tempfile.TemporaryDirectory() as tmp, FakeServer(respond) as server:
            task_dir = Path(tmp) / "tool_router"
            task_dir.mkdir()
            for name in ("task.json", "gold.jsonl"):  # not the live candidates/teacher/to_label files
                shutil.copy(TASK_DIR / name, task_dir / name)
            url = server.url + "/v1"
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["generate", str(task_dir), "--url", url, "--model", "gen", "--n", "5", "--seed", "3"]), 0)
                self.assertEqual(main(["blindcheck", str(task_dir), "--url", url, "--checkers", "agree,contrarian",
                                       "--tiebreaker", "agree"]), 0)
                self.assertEqual(main(["blindcheck", str(task_dir), "--url", url]), 1)  # nothing left
            self.assertEqual(len(read_jsonl(task_dir / "candidates.jsonl")), 3)
            sorted_ids = [r["id"] for r in read_jsonl(task_dir / "teacher.jsonl") + read_jsonl(task_dir / "to_label.jsonl")]
            self.assertEqual(len(sorted_ids), 3)
            self.assertIn("3 candidates checked", out.getvalue())


if __name__ == "__main__":
    unittest.main()
