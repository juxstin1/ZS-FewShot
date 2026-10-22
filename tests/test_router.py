import json
import unittest

from tests.fakes import FakeServer
from zsbench.router import Router


def replies(*contents):
    """Fake OpenAI-compatible server that answers with the given contents in order."""
    queue = list(contents)

    def respond(path, payload):
        content = queue.pop(0) if queue else json.dumps({"label": "web_search"})
        if content == "500":
            return 500, {"error": "boom"}
        return 200, {"choices": [{"message": {"role": "assistant", "content": content}}]}

    return respond


class RouterTest(unittest.TestCase):
    def router(self, server, **kw):
        return Router(url=server.url + "/v1", model="m", api_key="sk-test", load_env=False, **kw)

    def test_routes_and_reports_metadata(self):
        with FakeServer(replies(json.dumps({"label": "run_code"}))) as server:
            d = self.router(server).route("how many days between two dates", ["user: hi"])
        self.assertEqual((d.label, d.ok, d.attempts, d.wording_version), ("run_code", True, 1, "v1"))
        body, headers = server.requests[0]["body"], server.requests[0]["headers"]
        self.assertEqual(headers["Authorization"], "Bearer sk-test")
        self.assertIn("Previous turns:\n- user: hi", body["messages"][-1]["content"])
        self.assertNotIn("chat_template_kwargs", body)  # DeepInfra gets no vLLM-only fields

    def test_retries_then_succeeds(self):
        with FakeServer(replies("not json", json.dumps({"label": "ask_user"}))) as server:
            d = self.router(server).route("fix it")
        self.assertEqual((d.label, d.ok, d.attempts), ("ask_user", True, 2))

    def test_falls_back_after_retries(self):
        with FakeServer(replies("500", "500")) as server:
            d = self.router(server, fallback="answer_directly").route("hi")
        self.assertEqual((d.label, d.ok, d.attempts), ("answer_directly", False, 2))
        self.assertIn("500", d.error)

    def test_rejects_unknown_fallback(self):
        with self.assertRaises(ValueError):
            Router(url="http://x", api_key="k", load_env=False, fallback="panic")


if __name__ == "__main__":
    unittest.main()
