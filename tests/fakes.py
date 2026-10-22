"""In-process fake Simple Jev and OpenAI-compatible servers, shaped like the documented APIs."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeServer:
    """Runs a handler on 127.0.0.1:<free port>. Use as a context manager; .url, .requests (POSTs), .gets."""

    def __init__(self, respond):
        self.requests: list[dict] = []
        self.gets: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, status, body):
                data = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                outer.gets.append({"path": self.path, "headers": dict(self.headers)})
                self._send(200, {"status": "ok", "data": []})

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "body": payload, "headers": dict(self.headers)})
                self._send(*respond(self.path, payload))

            def log_message(self, format, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def jev_respond(path, payload):
    """Position-biased scorer: the first-listed label gets +3, a label named in the state gets +2."""
    answers = {}
    for qid, q in payload["questions"].items():
        candidates = list(q["criteria"])
        weights = {c: 1.0 + (3.0 if i == 0 else 0.0) + (2.0 if c in payload["state"] else 0.0) for i, c in enumerate(candidates)}
        total = sum(weights.values())
        probs = {c: w / total for c, w in weights.items()}
        choice = max(probs, key=lambda c: probs[c])
        answers[qid] = {"choice": choice, "confidence": probs[choice], "probabilities": probs}
    return 200, {"model": payload["model"], "answers": answers, "usage": {"input_tokens": 1, "output_tokens": 0}}


def llm_respond(path, payload):
    """Echoes the first allowed label named in the user message; 'garbage' gets non-JSON back."""
    text = payload["messages"][-1]["content"]
    allowed = payload["response_format"]["json_schema"]["schema"]["properties"]["label"]["enum"]
    if "garbage" in text:
        content = "Sure! The answer is web_search."
    else:
        label = next((a for a in allowed if a in text), allowed[0])
        content = json.dumps({"label": label})
    return 200, {"choices": [{"message": {"role": "assistant", "content": content}}]}
