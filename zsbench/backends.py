"""The baselines. Each predict(text, context) returns {"pred", "confidence"?, "probs"?, "latency_ms"?, "extra"?}.

context is the earlier conversation turns (oldest first), or an empty list for a first message.
pred is None when a backend produced no valid label (parse failure, server error).
latency_ms, when present, replaces the harness's own timing (replies collected elsewhere).
"""

import json
import re
import urllib.error
import urllib.request
from pathlib import Path

from .task import Task, read_jsonl, row_key


def menu_prompt(task: Task) -> str:
    """The question plus every allowed label with its description."""
    label_lines = "\n".join(f"- {label}: {desc}" for label, desc in task.labels.items())
    return f"{task.question}\n\nAllowed labels:\n{label_lines}"


def request_text(text: str, context: list[str] | None) -> str:
    if not context:
        return text
    turns = "\n".join(f"- {turn}" for turn in context)
    return f"Previous turns:\n{turns}\n\nCurrent request:\n{text}"


def chat_prompt(task: Task, text: str, context: list[str] | None = None) -> str:
    """One self-contained message for a chat app (e.g. Snapchat My AI). The full label menu goes
    in every message: chat bots drift from options given earlier in the thread."""
    body = request_text(text, context) if context else f"Current request:\n{text}"
    return f"{menu_prompt(task)}\n\n{body}\n\nReply with only the label, nothing else."


def parse_label(reply: str, labels) -> str | None:
    """The one allowed label a free-text reply names, or None if it names none or several.
    Case, quotes, backticks, and spaces or hyphens for underscores ("Run code") are all fine."""
    norm = re.sub(r"[\s\-]+", "_", reply.lower())
    found = {label for label in labels if re.search(rf"(?<![a-z]){re.escape(label)}(?![a-z])", norm)}
    return found.pop() if len(found) == 1 else None


class BackendError(RuntimeError):
    pass


def request_json(url: str, payload: dict | None, timeout: float, headers: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        raise BackendError(f"{url} -> HTTP {e.code}: {body}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise BackendError(f"{url} unreachable: {getattr(e, 'reason', e)}") from e


class RulesBackend:
    """Keyword/regex rules from task.json. First matching rule wins, else default_label."""

    name = "rules"

    def __init__(self, task: Task):
        self.default = task.default_label
        self.rules = [(r["label"], [re.compile(p, re.IGNORECASE) for p in r["patterns"]]) for r in task.rules]

    def check(self) -> None:
        pass

    def describe(self) -> dict:
        return {"rules": len(self.rules), "default_label": self.default}

    def predict(self, text: str, context: list[str] | None = None) -> dict:
        # Rules only see the latest message; that blind spot is part of the baseline.
        for label, patterns in self.rules:
            if any(p.search(text) for p in patterns):
                return {"pred": label}
        return {"pred": self.default}


class JevBackend:
    """Any server speaking the Jev System One API (`POST /v1/systemone`): Simple Jev, Kev, or
    the Intern-Decision wrapper in scripts/intern_server.py. Answers are label probabilities,
    not generated text.

    Option order can bias these models (Simple Jev, for one, assigns letters A, B, C... in
    criteria order), so each request asks the same question once per cyclic rotation of the
    label list. Every label sits in every position once; servers share the prefill, so the
    extra questions are cheap. pred is the argmax of the averaged distribution;
    extra.first_order is the answer with labels in task.json order, i.e. a naive single call.
    """

    def __init__(self, task: Task, url: str, model: str, orders: int = 0, timeout: float = 60, name: str = "jev"):
        labels = list(task.labels)
        n = len(labels) if orders <= 0 else min(orders, len(labels))
        self.orders = [labels[i:] + labels[:i] for i in range(n)]
        self.base = url.rstrip("/")
        self.model = model
        self.name = name
        self.question = task.question
        self.labels = task.labels
        self.timeout = timeout

    def check(self) -> None:
        request_json(f"{self.base}/v1/models", None, 10)

    def describe(self) -> dict:
        return {"url": self.base, "model": self.model, "orders": len(self.orders)}

    def predict(self, text: str, context: list[str] | None = None) -> dict:
        state = {"previous_turns": context, "current_request": text} if context else text
        questions = {
            f"order{i}": {
                "type": "choice",
                "instructions": self.question,
                "criteria": {label: self.labels[label] for label in order},
            }
            for i, order in enumerate(self.orders)
        }
        payload = {"model": self.model, "state": state, "questions": questions}
        try:
            answers = request_json(f"{self.base}/v1/systemone", payload, self.timeout)["answers"]
            avg = dict.fromkeys(self.labels, 0.0)
            choices = []
            for qid in questions:
                choices.append(answers[qid]["choice"])
                for label, p in answers[qid]["probabilities"].items():
                    avg[label] += p / len(questions)
        except BackendError as e:
            return {"pred": None, "extra": {"error": str(e)}}
        except (KeyError, TypeError) as e:
            return {"pred": None, "extra": {"error": f"unexpected response shape: {e!r}"}}
        pred = max(avg, key=lambda label: avg[label])
        return {
            "pred": pred,
            "confidence": avg[pred],
            "probs": avg,
            "extra": {
                "order_choices": choices,
                "first_order": choices[0],
                "first_order_confidence": answers["order0"]["confidence"],
            },
        }


class LLMBackend:
    """OpenAI-compatible /chat/completions with a JSON-schema response_format (vLLM, llama.cpp, DeepInfra).

    Point it at the same open model Simple Jev serves, so the comparison isolates
    scoring labels vs generating them. api_key is sent as a Bearer token when set.
    """

    name = "llm"

    def __init__(
        self,
        task: Task,
        url: str,
        model: str,
        extra: dict | None = None,
        timeout: float = 60,
        api_key: str | None = None,
        max_tokens: int = 1024,  # a cap, not a target: reasoning models spend tokens before the JSON
    ):
        self.system = f'{menu_prompt(task)}\n\nReply with JSON only: {{"label": "<one allowed label>"}}'
        self.schema = {
            "type": "object",
            "properties": {"label": {"type": "string", "enum": list(task.labels)}},
            "required": ["label"],
            "additionalProperties": False,
        }
        self.base = url.rstrip("/")
        self.model = model
        self.labels = task.labels
        self.extra = extra or {}
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def check(self) -> None:
        request_json(f"{self.base}/models", None, 10, self.headers)

    def describe(self) -> dict:
        return {"url": self.base, "model": self.model, "extra": self.extra}

    def predict(self, text: str, context: list[str] | None = None) -> dict:
        text = request_text(text, context)
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": self.system}, {"role": "user", "content": text}],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "decision", "strict": True, "schema": self.schema},
            },
            **self.extra,
        }
        try:
            content = request_json(f"{self.base}/chat/completions", payload, self.timeout, self.headers)["choices"][0]["message"]["content"]
        except BackendError as e:
            return {"pred": None, "extra": {"error": str(e)}}
        try:
            label = json.loads(content)["label"]
        except (json.JSONDecodeError, KeyError, TypeError):
            label = None
        if not isinstance(label, str) or label not in self.labels:
            return {"pred": None, "extra": {"raw": (content or "")[:200]}}
        return {"pred": label}


class AnswersBackend:
    """Replies collected outside the harness, e.g. typed into a chat app from `zsbench prompts`.

    The answers file is that prompts file with "reply" filled in (and optionally "latency_ms").
    Rows are matched by "key" (task.row_key), so ids can't drift. A row with no reply, or a
    reply that names no single allowed label, scores invalid.
    """

    def __init__(self, task: Task, path: str, model: str, name: str):
        self.path = Path(path)
        self.model = model
        self.name = name
        self.labels = task.labels
        self.replies: dict[str, dict] = {}

    def check(self) -> None:
        if not self.path.exists():
            raise BackendError(f"{self.path} not found")
        self.replies = {r["key"]: r for r in read_jsonl(self.path) if r.get("reply")}

    def describe(self) -> dict:
        return {"answers": str(self.path), "model": self.model, "replied": len(self.replies)}

    def predict(self, text: str, context: list[str] | None = None) -> dict:
        row = self.replies.get(row_key({"text": text, "context": context or []}))
        if row is None:
            return {"pred": None, "latency_ms": None, "extra": {"error": "no reply collected"}}
        return {
            "pred": parse_label(row["reply"], self.labels),
            "latency_ms": row.get("latency_ms"),
            "extra": {"raw": row["reply"][:200]},
        }
