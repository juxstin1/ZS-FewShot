"""Synthetic candidates: generators write hard inputs with an intended label, blind checkers vote.

Agreed rows become teacher-labeled training data. Disputed rows go to a human for hand-labeling.
Rows from one contrastive pair share a `group` and always land in the same pile, so near-copies
never straddle training data and the gold set. Mix generators (and styles) so a classifier learns
the routing signal rather than one model's phrasing habits.
"""

import hashlib
import itertools
import json
import random
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from .backends import BackendError, LLMBackend, request_json
from .task import Task, context_turns, near_copies, row_key

KINDS = {
    "clear": "Clear-cut, everyday requests where the correct action is obvious to anyone. Realistic volume "
    "data, not trick questions.",
    "near_miss": "The wording points at one action, but the correct action is a different one "
    "(it mentions a file but really needs current web info, or sounds like a question but needs code to run).",
    "needs_context": "A short follow-up such as 'yes do it' or 'try the other one' that can only be labeled "
    "with the 1-3 earlier turns you also write.",
    "vague": "Underspecified requests where the agent must ask the user first, mixed with short requests "
    "that look vague but are actually answerable as written.",
    "contrastive_pair": "Pairs of nearly identical requests (change at most 8 words) where the small change "
    "flips the correct action. Give both halves of a pair the same non-empty pair_id.",
    "multi_step": "Multi-step asks (do X, then Y, then Z) where exactly one action has to come first; "
    "`intended` is that first action.",
    "messy": "Realistic messy messages: typos, slang, run-on sentences, or two asks in one where one "
    "clearly decides the next action.",
    "boundary": "Boundary cases between two specific actions (given below), where only the exact action "
    "definitions settle the answer.",
}

# Kinds meant to be hard; their prompts forbid giveaway keywords.
EDGE_KINDS = {"near_miss", "needs_context", "vague", "contrastive_pair", "multi_step", "boundary"}

# (correct, tempting) action pairs for the boundary kind, from the checkers' most common confusions.
BOUNDARY_PAIRS = [
    ("web_search", "answer_directly"), ("answer_directly", "web_search"),
    ("ask_user", "answer_directly"), ("answer_directly", "ask_user"),
    ("run_code", "read_file"), ("read_file", "run_code"),
    ("read_file", "ask_user"), ("ask_user", "read_file"),
    ("web_search", "read_file"), ("run_code", "answer_directly"),
]

TOPICS = [
    "ML training runs", "personal finance spreadsheets", "web app debugging", "travel planning",
    "home lab servers", "video editing", "contracts and legal documents", "pandas data analysis",
    "music production", "job applications", "homework help", "CI pipelines and deploys",
    "fitness tracking", "reading research papers", "small business invoicing", "smart home automation",
]

STYLES = [
    "a terse senior engineer who types lowercase fragments",
    "a non-native English speaker with small grammar slips",
    "a busy parent on a phone: typos, no punctuation",
    "a polite, wordy office worker who over-explains",
    "a student using slang and abbreviations",
    "a small-business owner who is a bit frustrated",
    "a researcher writing long, precise, multi-sentence requests",
    "someone dictating by voice: run-on sentences and filler words",
]

SYSTEM = (
    "You write evaluation data for an AI agent's tool router. The router reads a user message "
    "(plus any earlier turns) and picks the agent's single next action. Every example must have "
    "exactly one correct action that a careful human would agree on, following the action "
    "definitions exactly."
)


class GenerationError(RuntimeError):
    pass


def make_id(row: dict) -> str:
    return hashlib.sha1(row_key(row).encode("utf-8")).hexdigest()[:12]


def examples_schema(labels: list[str], boundary: bool = False) -> dict:
    """Boundary batches also return `rival` (the tempting wrong action) and `why` (which definition
    detail decides it). Both stay out of anything a checker sees."""
    properties = {
        "text": {"type": "string"},
        "context": {"type": "array", "items": {"type": "string"}},
        "intended": {"type": "string", "enum": labels},
        "pair_id": {"type": "string"},
    }
    if boundary:
        properties |= {"rival": {"type": "string", "enum": labels}, "why": {"type": "string"}}
    return {
        "type": "object",
        "properties": {
            "examples": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties),
                    "additionalProperties": False,
                },
            }
        },
        "required": ["examples"],
        "additionalProperties": False,
    }


def build_prompt(labels: dict[str, str], kind: str, topic: str, k: int, targets: list[str], style: str) -> str:
    menu = "\n".join(f"- {label}: {desc}" for label, desc in labels.items())
    if kind == "boundary":
        correct, rival = targets[0], targets[1]
        aim = (
            f"Every example sits right on the line between {correct} and {rival}: a quick reader would "
            f"pick {rival}, but applying the definitions exactly makes {correct} correct. Set `intended` "
            f"to {correct}, `rival` to {rival}, and `why` to the one definition detail that decides it."
        )
    else:
        aim = f"Mostly aim the correct action at: {', '.join(targets)}."
    hard = (
        "Make these genuinely hard. No giveaway keywords ('latest', 'run', 'search', a file name) unless "
        "they point the wrong way; the deciding detail should be subtle but real. Still exactly one "
        "correct action under the definitions.\n"
        if kind in EDGE_KINDS else ""
    )
    return (
        f"Actions the router can pick:\n{menu}\n\n"
        f"Write {k} examples of this kind: {KINDS[kind]}\n"
        f"Topic area: {topic}. Write them as {style}. Mix lengths: some under 8 words, some 1-3 "
        f"sentences, an occasional long one. {aim}\n{hard}\n"
        "Fields: `text` is the latest user message. `context` lists earlier turns, oldest first, "
        "each starting with 'user: ' or 'assistant: ' (an empty list when there are none). "
        "`intended` is the one correct action. `pair_id` is empty unless the example is half of a "
        "contrastive pair. Don't number anything."
    )


def rows_from_examples(examples: list[dict], labels: dict[str, str], batch_id: str, model: str,
                       **meta) -> list[dict]:
    rows = []
    for ex in examples:
        text = str(ex.get("text") or "").strip()
        if not text or ex.get("intended") not in labels:
            continue
        row: dict = {"text": text}
        context = [t.strip() for t in ex.get("context") or [] if isinstance(t, str) and t.strip()]
        if context:
            row["context"] = context
        row["id"] = make_id(row)
        pair = str(ex.get("pair_id") or "").strip()
        extra = {k: ex[k] for k in ("rival", "why") if ex.get(k)}
        row.update({
            "intended": ex["intended"],
            **extra,
            **meta,
            "group": f"{batch_id}:{pair}" if pair else row["id"],
            "generator": model,
        })
        rows.append(row)
    return rows


class Generator:
    """An OpenAI-compatible chat model (DeepInfra, vLLM) under a JSON schema."""

    def __init__(self, task: Task, url: str, model: str, api_key: str | None = None,
                 temperature: float = 0.9, timeout: float = 300):
        self.labels = task.labels
        self.base = url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def check(self) -> None:
        request_json(f"{self.base}/models", None, 10, self.headers)

    def batch(self, kind: str, topic: str, k: int, targets: list[str], batch_id: str, style: str) -> list[dict]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_prompt(self.labels, kind, topic, k, targets, style)},
            ],
            "temperature": self.temperature,
            "max_tokens": 8192,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "candidates",
                    "strict": True,
                    "schema": examples_schema(list(self.labels), boundary=kind == "boundary"),
                },
            },
        }
        resp = request_json(f"{self.base}/chat/completions", payload, self.timeout, self.headers)
        examples = json.loads(resp["choices"][0]["message"]["content"])["examples"]
        return rows_from_examples(examples, self.labels, batch_id, self.model, kind=kind, topic=topic, style=style)


class ClaudeGenerator:
    """Claude through the official SDK, with structured output. Haiku 5.5 for bulk, Sonnet 5.5 for
    edge cases. The SDK reads ANTHROPIC_API_KEY. Diversity comes from styles and topics: the API no
    longer takes temperature."""

    def __init__(self, task: Task, model: str, effort: str = "low", timeout: float = 300, client=None):
        import anthropic

        self.sdk = anthropic
        self.client = client or anthropic.Anthropic(timeout=timeout)
        self.labels = task.labels
        self.model = model
        self.effort = effort

    def check(self) -> None:
        try:
            self.client.models.retrieve(self.model)
        except self.sdk.AuthenticationError as e:
            raise GenerationError("Claude API key rejected: check ANTHROPIC_API_KEY in .env") from e
        except self.sdk.NotFoundError as e:
            raise GenerationError(f"unknown Claude model {self.model!r}") from e
        except self.sdk.APIError as e:
            raise GenerationError(f"Claude API check failed: {e}") from e

    def batch(self, kind: str, topic: str, k: int, targets: list[str], batch_id: str, style: str) -> list[dict]:
        prompt = build_prompt(self.labels, kind, topic, k, targets, style)
        data = self.ask_json(SYSTEM, prompt, examples_schema(list(self.labels), boundary=kind == "boundary"))
        return rows_from_examples(data["examples"], self.labels, batch_id, self.model, kind=kind, topic=topic,
                                  style=style)

    def ask_json(self, system: str, prompt: str, schema: dict) -> dict:
        """One structured-output call. Raises GenerationError on API errors, refusals and truncation."""
        messages = [{"role": "user", "content": prompt}]
        output_config = {"effort": self.effort, "format": {"type": "json_schema", "schema": schema}}
        try:
            if self.model.startswith("claude-haiku"):
                # Haiku has no server-side fallback.
                response = self.client.messages.create(
                    model=self.model, max_tokens=16000, system=system, messages=messages, output_config=output_config
                )
            else:
                # On a safety decline, the API re-runs the request on Anthropic's recommended fallback.
                response = self.client.beta.messages.create(
                    model=self.model, max_tokens=16000, system=system, messages=messages, output_config=output_config,
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                )
        except self.sdk.RateLimitError as e:
            raise GenerationError(f"rate limited: {e}") from e
        except self.sdk.APIStatusError as e:
            raise GenerationError(f"HTTP {e.status_code}: {e.message}") from e
        except self.sdk.APIConnectionError as e:
            raise GenerationError(f"connection failed: {e}") from e
        if response.stop_reason in ("refusal", "max_tokens"):
            raise GenerationError(f"stopped: {response.stop_reason}")
        text = next(b.text for b in response.content if b.type == "text")
        return json.loads(text)


ADJUDICATOR_SYSTEM = (
    "You settle disputes for an AI agent's tool router. Blind labelers disagreed about the agent's single "
    "next action for a user message. Apply the action definitions exactly, not your sense of the topic: "
    "pick the action the request asks for."
)


class ClaudeAdjudicator(ClaudeGenerator):
    """Settles a disputed row by choosing among the labels the generator and checkers proposed.
    Its output is training data only: a model's call never goes into the gold set."""

    def decide(self, row: dict) -> tuple[str, str]:
        order = list(self.labels)
        proposed = {row["intended"], *[v for v in row.get("votes", {}).values() if v in self.labels]}
        options = sorted(proposed, key=order.index)
        menu = "\n".join(f"- {label}: {self.labels[label]}" for label in order)
        turns = "\n".join(f"  {t}" for t in context_turns(row))
        prompt = (
            f"Action definitions:\n{menu}\n\n"
            + (f"Earlier turns:\n{turns}\n\n" if turns else "")
            + f"Latest user message:\n{row['text']}\n\n"
            f"The labelers split between: {', '.join(options)}. Which one is correct under the definitions? "
            "`why` is one sentence naming the deciding detail."
        )
        schema = {
            "type": "object",
            "properties": {"label": {"type": "string", "enum": options}, "why": {"type": "string"}},
            "required": ["label", "why"],
            "additionalProperties": False,
        }
        data = self.ask_json(ADJUDICATOR_SYSTEM, prompt, schema)
        return data["label"], data["why"]


def adjudicate(task: Task, hand: list[dict], judge: ClaudeAdjudicator, workers: int = 4,
               log: Callable[[str], None] = print) -> tuple[list[dict], list[dict], list[dict]]:
    """Split the hand-label queue into (settled, still_open, done).

    Rows already in gold are done. A group with any gold row stays open for a human, so pair halves never
    straddle gold and training data. The rest go to the judge; failures stay open."""
    gold_ids = {g["id"] for g in task.gold}
    gold_groups = {r["group"] for r in hand if r["id"] in gold_ids}
    done = [r for r in hand if r["id"] in gold_ids]
    still_open = [r for r in hand if r["id"] not in gold_ids and r["group"] in gold_groups]
    todo = [r for r in hand if r["id"] not in gold_ids and r["group"] not in gold_groups]

    def one(r):
        try:
            return r, *judge.decide(r)
        except (GenerationError, json.JSONDecodeError, KeyError, StopIteration) as e:
            log(f"{r['id']}: {e}")
            return r, None, None

    settled = []
    with ThreadPoolExecutor(workers) as pool:
        for r, label, why in pool.map(one, todo):
            if label is None:
                still_open.append(r)
                continue
            base = {k: r[k] for k in ("id", "text", "context") if k in r}
            settled.append({
                **base, "label": label, "source": "teacher", "agreement": "adjudicated",
                "adjudicator": judge.model, "adjudication_why": why,
                **{k: r[k] for k in ("intended", "votes", "kind", "group", "generator", "rival", "why") if k in r},
            })
    return settled, still_open, done


def make_generator(task: Task, model: str, url: str, api_key: str | None, temperature: float = 0.9,
                   effort: str | None = None):
    if model.startswith("claude-"):
        return ClaudeGenerator(task, model, effort=effort or ("low" if "haiku" in model else "medium"))
    return Generator(task, url, model, api_key=api_key, temperature=temperature)


def _groups(rows: list[dict]) -> list[list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    return list(groups.values())


def generate(task: Task, gen, n: int, batch: int = 8, existing: list[dict] | None = None,
             seed: int | None = None, workers: int = 4, kinds: list[str] | None = None,
             log: Callable[[str], None] = print) -> list[dict]:
    """Collect about n new candidates, dropping any group that near-copies gold or earlier candidates."""
    rng = random.Random(seed)
    labels = list(task.labels)
    kinds = list(kinds or KINDS)
    rng.shuffle(kinds)  # then round-robin, so small runs still cover every requested kind
    pairs = [p for p in BOUNDARY_PAIRS if p[0] in task.labels and p[1] in task.labels] or [
        tuple(rng.sample(labels, 2)) for _ in range(len(labels))
    ]
    rng.shuffle(pairs)
    pair_cycle = itertools.cycle(pairs)
    seen = [row_key(r) for r in [*task.gold, *(existing or [])]]
    out: list[dict] = []
    calls, max_calls = 0, max(3, 3 * -(-n // batch))
    run_tag = f"{rng.getrandbits(32):08x}"

    def targets_for(kind: str) -> list[str]:
        if kind == "boundary":
            return list(next(pair_cycle))
        return rng.sample(labels, min(3, len(labels)))

    def one(job):
        kind, topic, targets, batch_id, style = job
        try:
            return gen.batch(kind, topic, batch, targets, batch_id, style)
        except (BackendError, GenerationError, json.JSONDecodeError, KeyError, TypeError, IndexError,
                StopIteration) as e:
            log(f"batch {batch_id} ({kind}, {topic}) failed: {e}")
            return []

    while len(out) < n and calls < max_calls:
        jobs = []
        for _ in range(min(workers, max_calls - calls)):
            kind = kinds[calls % len(kinds)]
            jobs.append((kind, rng.choice(TOPICS), targets_for(kind), f"{run_tag}-{calls}", rng.choice(STYLES)))
            calls += 1
        with ThreadPoolExecutor(len(jobs)) as pool:
            results = list(pool.map(one, jobs))
        for rows in results:
            for group in _groups(rows):
                keys = [row_key(r) for r in group]
                if len(out) >= n or near_copies(keys, seen):
                    continue
                seen.extend(keys)
                out.extend(group)
        log(f"generated {len(out)}/{n} after {calls} calls")
    return out


def vote(intended: str, checker_votes: list[str | None],
         tiebreak: Callable[[], str | None] | None = None) -> tuple[str, str | None, str | None]:
    """Return (pile, agreement, tiebreak_vote). pile is "teacher" or "hand".

    All checkers match the intended label: teacher (unanimous). An even split: the tiebreaker
    decides. A strict majority: teacher (majority). Anything else, including checkers agreeing
    with each other against the generator, goes to a human.
    """
    n = len(checker_votes)
    matches = sum(v == intended for v in checker_votes)
    if matches == n:
        return "teacher", "unanimous", None
    if tiebreak and matches * 2 == n:
        tb = tiebreak()
        return ("teacher", "tiebreak", tb) if tb == intended else ("hand", None, tb)
    if matches * 2 > n:
        return "teacher", "majority", None
    return "hand", None, None


def blindcheck(candidates: list[dict], checkers: list[LLMBackend], tiebreaker: LLMBackend | None = None,
               workers: int = 8, log: Callable[[str], None] = print) -> tuple[list[dict], list[dict], dict]:
    """Checkers see only the input and the label menu, never `intended`."""
    models = [chk.model for chk in checkers]
    if len(set(models)) != len(models):
        raise ValueError(f"checker models must be distinct, got {models}")

    def one(c):
        context = context_turns(c)
        votes = {chk.model: chk.predict(c["text"], context)["pred"] for chk in checkers}
        asked = {}

        def tiebreak():
            assert tiebreaker is not None
            asked[f"tiebreak:{tiebreaker.model}"] = tiebreaker.predict(c["text"], context)["pred"]
            return next(iter(asked.values()))

        pile, agreement, _ = vote(c["intended"], list(votes.values()), tiebreak if tiebreaker else None)
        return c, pile, agreement, {**votes, **asked}

    results = []
    with ThreadPoolExecutor(workers) as pool:
        for i, result in enumerate(pool.map(one, candidates), 1):
            results.append(result)
            if i % 10 == 0 or i == len(candidates):
                log(f"checked {i}/{len(candidates)}")

    disputed_groups = {c["group"] for c, pile, _, _ in results if pile == "hand"}
    teacher, hand = [], []
    stats = Counter()
    per_checker = Counter()
    invalid = Counter()
    for c, pile, agreement, votes in results:
        for chk in checkers:
            per_checker[chk.model] += votes[chk.model] == c["intended"]
            invalid[chk.model] += votes[chk.model] is None
        base = {k: c[k] for k in ("id", "text", "context") if k in c}
        meta = {"votes": votes, "kind": c.get("kind"), "group": c["group"], "generator": c.get("generator"),
                **{k: c[k] for k in ("rival", "why") if k in c}}
        if pile == "teacher" and c["group"] not in disputed_groups:
            teacher.append({**base, "label": c["intended"], "source": "teacher", "agreement": agreement, **meta})
            stats[agreement] += 1
        else:
            hand.append({**base, "intended": c["intended"], **meta})
            stats["pulled_by_pair" if pile == "teacher" else "disputed"] += 1
    summary = {
        "n": len(results),
        "piles": dict(stats),
        "checker_agrees_with_generator": {m: per_checker[m] / len(results) for m in per_checker} if results else {},
        "checker_invalid": dict(invalid),
    }
    return teacher, hand, summary


def take_whole_groups(rows: list[dict], limit: int) -> list[dict]:
    """The first rows up to about `limit`, never splitting a group."""
    if not limit:
        return rows
    keep_groups, count = set(), 0
    for row in rows:
        if row["group"] not in keep_groups:
            if count >= limit:
                continue
            keep_groups.add(row["group"])
        count += 1
    return [r for r in rows if r["group"] in keep_groups]
