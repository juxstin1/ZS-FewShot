"""Serve Intern-Decision's DecisionEngine over the Jev System One API, so zsbench can score it.

Intern-Decision ships a Python DecisionEngine but no server. This wraps it:
  GET  /v1/models     -> {"data": [{"id": "intern-decision-4b"}]}
  POST /v1/systemone  -> engine.predict({"state", "questions"[, "images"]})

Run it with the Intern venv (torch + transformers from the model's requirements.txt):
  servers/intern-venv/Scripts/python.exe scripts/intern_server.py servers/intern-decision-4b --port 8010
"""

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gpu_guard import LOCKOUT  # noqa: E402 - same scripts/ folder

    if LOCKOUT:
        sys.exit(f"intern_server: REFUSED: {LOCKOUT}")

    model_dir = Path(args.model_dir).resolve()
    sys.path.insert(0, str(model_dir))
    from inference import DecisionEngine  # noqa: E402 - lives in the model repo

    engine = DecisionEngine(checkpoint=str(model_dir), device="cuda")
    lock = threading.Lock()  # one GPU, one request at a time
    model_id = model_dir.name

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                self._send(200, {"data": [{"id": model_id}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/systemone":
                self._send(404, {"error": "not found"})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                request = {k: body[k] for k in ("state", "questions", "images") if k in body}
                with lock:
                    response = engine.predict(request)
                self._send(200, response)
            except (ValueError, KeyError, TypeError) as e:  # bad request shapes and over-limit inputs
                self._send(422, {"error": str(e)})

        def log_message(self, format, *args):
            pass

    print(f"intern-decision serving {model_id} on http://{args.host}:{args.port}", flush=True)
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
