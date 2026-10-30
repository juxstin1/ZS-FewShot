import hashlib
import json
import unittest
from pathlib import Path

from tests.fakes import FakeServer, jev_respond, llm_respond
from zsbench.backends import BackendError, JevBackend, LLMBackend, RulesBackend, chat_prompt, parse_label
from zsbench.task import load_task, validate

TASK_DIR = Path(__file__).resolve().parent.parent / "tasks" / "tool_router"


class ToolRouterTaskTest(unittest.TestCase):
    def test_shipped_task_is_valid(self):
        task = load_task(TASK_DIR)
        self.assertEqual(validate(task), [])
        self.assertEqual(sum(r.get("source") == "seed" for r in task.gold), 22)


class WordingFreezeTest(unittest.TestCase):
    def test_wording_is_frozen_at_v1(self):
        cfg = json.loads((TASK_DIR / "task.json").read_text(encoding="utf-8"))
        blob = json.dumps({"question": cfg["question"], "labels": cfg["labels"]}, sort_keys=True)
        fingerprint = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
        self.assertEqual(cfg.get("wording_version"), "v1")
        self.assertEqual(
            fingerprint, "770dd97d3a97806e",
            "tool_router wording is frozen at v1 (README changelog). Grow the gold set instead; if the "
            "wording really must change, add a v2 changelog entry and update this fingerprint.",
        )


class RulesTest(unittest.TestCase):
    def test_first_matching_rule_wins_else_default(self):
        rules = RulesBackend(load_task(TASK_DIR))
        self.assertEqual(rules.predict("Summarize notes.md for me please")["pred"], "read_file")
        self.assertEqual(rules.predict("What's the latest Qwen release?")["pred"], "web_search")
        self.assertEqual(rules.predict("Fix it.")["pred"], "ask_user")
        self.assertEqual(rules.predict("Explain how attention works in a transformer.")["pred"], "answer_directly")


class JevTest(unittest.TestCase):
    def test_rotations_cancel_position_bias(self):
        task = load_task(TASK_DIR)
        with FakeServer(jev_respond) as server:
            jev = JevBackend(task, server.url, "m")
            jev.check()
            result = jev.predict("please web_search this")

        body = server.requests[0]["body"]
        self.assertEqual(server.requests[0]["path"], "/v1/systemone")
        firsts = [next(iter(q["criteria"])) for q in body["questions"].values()]
        self.assertEqual(sorted(firsts), sorted(task.labels))  # every label leads once
        self.assertEqual(result["extra"]["first_order"], "answer_directly")  # biased single call
        self.assertEqual(result["pred"], "web_search")  # averaged over orders
        self.assertAlmostEqual(sum(result["probs"].values()), 1.0)

    def test_context_goes_into_state(self):
        with FakeServer(jev_respond) as server:
            JevBackend(load_task(TASK_DIR), server.url, "m").predict("yes, do it", ["assistant: rerun the tests?"])
            JevBackend(load_task(TASK_DIR), server.url, "m").predict("hi", [])
        self.assertEqual(
            server.requests[0]["body"]["state"],
            {"previous_turns": ["assistant: rerun the tests?"], "current_request": "yes, do it"},
        )
        self.assertEqual(server.requests[1]["body"]["state"], "hi")

    def test_orders_one_sends_a_single_question(self):
        with FakeServer(jev_respond) as server:
            JevBackend(load_task(TASK_DIR), server.url, "m", orders=1).predict("hi")
        self.assertEqual(len(server.requests[0]["body"]["questions"]), 1)

    def test_server_error_becomes_invalid_prediction(self):
        with FakeServer(lambda path, payload: (422, {"detail": "label is not one token"})) as server:
            result = JevBackend(load_task(TASK_DIR), server.url, "m").predict("hi")
        self.assertIsNone(result["pred"])
        self.assertIn("422", result["extra"]["error"])

    def test_check_fails_fast_when_server_is_down(self):
        with self.assertRaises(BackendError):
            JevBackend(load_task(TASK_DIR), "http://127.0.0.1:9", "m").check()


class LLMTest(unittest.TestCase):
    def test_structured_output_request_and_parse(self):
        task = load_task(TASK_DIR)
        with FakeServer(llm_respond) as server:
            llm = LLMBackend(task, server.url + "/v1", "m", extra={"chat_template_kwargs": {"enable_thinking": False}})
            good = llm.predict("I need run_code for this")
            bad = llm.predict("garbage please")

        body = server.requests[0]["body"]
        self.assertEqual(server.requests[0]["path"], "/v1/chat/completions")
        self.assertEqual(body["response_format"]["json_schema"]["schema"]["properties"]["label"]["enum"], list(task.labels))
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(good["pred"], "run_code")
        self.assertIsNone(bad["pred"])
        self.assertIn("web_search", bad["extra"]["raw"])

    def test_api_key_sent_as_bearer_only_when_set(self):
        task = load_task(TASK_DIR)
        with FakeServer(llm_respond) as server:
            keyed = LLMBackend(task, server.url + "/v1", "m", api_key="sk-test")
            keyed.check()
            keyed.predict("hi")
            LLMBackend(task, server.url + "/v1", "m").predict("hi")
        self.assertEqual(server.gets[0]["headers"]["Authorization"], "Bearer sk-test")
        self.assertEqual(server.requests[0]["headers"]["Authorization"], "Bearer sk-test")
        self.assertNotIn("Authorization", server.requests[1]["headers"])
        self.assertNotIn("sk-test", str(keyed.describe()))

    def test_context_is_prepended_to_user_message(self):
        with FakeServer(llm_respond) as server:
            LLMBackend(load_task(TASK_DIR), server.url + "/v1", "m").predict("yes, do it", ["assistant: rerun the tests?"])
        user = server.requests[0]["body"]["messages"][-1]["content"]
        self.assertTrue(user.startswith("Previous turns:\n- assistant: rerun the tests?"))
        self.assertTrue(user.endswith("Current request:\nyes, do it"))


class ChatPromptTest(unittest.TestCase):
    def test_every_prompt_carries_the_full_menu(self):
        task = load_task(TASK_DIR)
        for prompt in (chat_prompt(task, "hi"), chat_prompt(task, "yes, do it", ["assistant: rerun the tests?"])):
            for label, desc in task.labels.items():
                self.assertIn(f"- {label}: {desc}", prompt)
            self.assertTrue(prompt.endswith("Reply with only the label, nothing else."))
        self.assertIn("Current request:\nhi\n", chat_prompt(task, "hi"))
        self.assertIn("Previous turns:\n- assistant: rerun the tests?", chat_prompt(task, "yes", ["assistant: rerun the tests?"]))

    def test_parse_label_takes_one_named_label_only(self):
        labels = load_task(TASK_DIR).labels
        self.assertEqual(parse_label("run_code", labels), "run_code")
        self.assertEqual(parse_label("`Run code`.", labels), "run_code")
        self.assertEqual(parse_label("Answer-directly", labels), "answer_directly")
        self.assertIsNone(parse_label("web search or read file", labels))
        self.assertIsNone(parse_label("meow", labels))


if __name__ == "__main__":
    unittest.main()
