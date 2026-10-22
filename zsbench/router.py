"""The agent's router: DeepSeek V4.1 Flash on DeepInfra picks the next action, using the frozen
tool_router v1 label wording. Chosen 2026-10-22: best on the BFCL routing set, and as an API it
puts no load on the PROD GPU.

    from zsbench.router import Router
    router = Router()                                   # reads .env; key from ROUTER_API_KEY or LLM_API_KEY
    d = router.route("what's the latest vllm release?")  # Decision(label='web_search', ok=True, ...)
    d = router.route("yes do it", ["assistant: Want me to rerun the tests?"])

A failed or unparseable call is retried, then returns `fallback` (answer_directly by default, so the
main agent model handles the turn itself) with ok=False and the error, so the agent never blocks on it.
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from .backends import BackendError, LLMBackend
from .env import load_dotenv
from .task import load_task

TASK_DIR = Path(__file__).resolve().parent.parent / "tasks" / "tool_router"
DEFAULT_URL = "https://api.deepinfra.com/v1/openai"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"


@dataclass
class Decision:
    label: str
    ok: bool
    latency_ms: float
    attempts: int
    model: str
    wording_version: str
    error: str | None = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class Router:
    def __init__(self, task_dir: str | Path = TASK_DIR, url: str | None = None, model: str | None = None,
                 api_key: str | None = None, timeout: float = 15, retries: int = 1,
                 fallback: str = "answer_directly", load_env: bool = True):
        if load_env:
            load_dotenv()
        task = load_task(task_dir)
        if fallback not in task.labels:
            raise ValueError(f"fallback {fallback!r} is not one of {list(task.labels)}")
        self.labels = list(task.labels)
        self.wording_version = json.loads((Path(task_dir) / "task.json").read_text(encoding="utf-8")).get(
            "wording_version", "unversioned")
        self.model = model or os.environ.get("ROUTER_MODEL", DEFAULT_MODEL)
        self.backend = LLMBackend(
            task,
            url or os.environ.get("ROUTER_URL", DEFAULT_URL),
            self.model,
            extra=json.loads(os.environ.get("ROUTER_EXTRA", "{}")),
            timeout=timeout,
            api_key=api_key or os.environ.get("ROUTER_API_KEY") or os.environ.get("LLM_API_KEY"),
        )
        self.retries = retries
        self.fallback = fallback

    def check(self) -> None:
        """Raise BackendError if the endpoint or key doesn't work."""
        self.backend.check()

    def route(self, text: str, context: list[str] | None = None) -> Decision:
        start = time.perf_counter()
        error = None
        for attempt in range(1, self.retries + 2):
            result = self.backend.predict(text, context or [])
            if result["pred"] is not None:
                return Decision(result["pred"], True, (time.perf_counter() - start) * 1000, attempt, self.model,
                                self.wording_version)
            extra = result.get("extra", {})
            error = extra.get("error") or f"unusable answer: {extra.get('raw', '')!r}"
        return Decision(self.fallback, False, (time.perf_counter() - start) * 1000, self.retries + 1, self.model,
                        self.wording_version, error)


__all__ = ["Router", "Decision", "BackendError"]
