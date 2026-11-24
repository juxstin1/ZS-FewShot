# zsbench: full operational details

Moved out of the README. Covers the router API, harness, gold-set rules, synthetic data pipeline, DeepInfra setup, label wording changelog and tests.

## The router (production)

[2026-10-22] The router is **DeepSeek V4.1 Flash on DeepInfra**, using the frozen v1 label wording. It was the best model on the BFCL routing set and puts no load on the PROD GPU. Results are in [docs/jev-landscape.md](docs/jev-landscape.md).

```python
from zsbench.router import Router

router = Router()  # reads .env: key from ROUTER_API_KEY or LLM_API_KEY
d = router.route("what's the latest vLLM release?")             # d.label == "web_search"
d = router.route("yes do it", ["user: the tests keep failing",
                               "assistant: Want me to rerun them with -x?"])  # "run_code"
if not d.ok:
    ...  # the call failed after a retry: d.label is the fallback ("answer_directly"), d.error says why
```

From a shell or another language, `route` prints one JSON decision and exits 1 when it fell back:

```bash
python -m zsbench route "summarize the README in this repo" --context "user: hi"
```

- **Labels:** `answer_directly`, `web_search`, `read_file`, `run_code`, `ask_user`.
- **Every decision logs** its label, ok, latency_ms, attempts, model and wording_version.
- **Latency:** 0.6–2.7 s per call in the live smoke test, with a 15 s timeout and 1 retry.
- **Overrides:** `ROUTER_MODEL`, `ROUTER_URL` and `ROUTER_EXTRA` in `.env`.
- **Training data:** to use real agent runs as training data later, log them in the format in [docs/trace-logging.md](docs/trace-logging.md). That's one JSONL file per day with `decision` and `run_end` lines, written up as typed schemas with examples, plus shadow mode for comparing the router against the agent.

## The benchmark harness

Baseline harness for one agent decision. It compares three ways of making the same call on the same gold set:

| backend | what it is |
|---|---|
| `rules` | Regex rules from `task.json`. The first rule that matches wins. |
| `jev` | Any Jev System One server ([Simple Jev](https://github.com/featherless-ai/simple-jev), Kev, Intern-Decision). It scores each label from next-token logits, with no generation. Never run locally here; see the lockout below. |
| `llm` | The same Qwen on vLLM, generating `{"label": ...}` under a JSON schema. |

`jev` and `llm` use the same model on purpose. The gap between them is the effect of scoring labels instead of generating them.

The harness itself is stdlib-only Python 3.11+. The one exception is the Claude generators, which need the `anthropic` SDK in `.venv`. The model servers run separately.

Data generation uses DeepInfra open models plus the Claude API: Haiku 5.5 for bulk, Sonnet 5.5 for edge cases ([2026-10-22] Justin's call; this reverses the earlier no-Claude rule). Mix generators so the classifier learns the routing signal, not one model's phrasing habits. See [docs/approach.md](docs/approach.md).

## The task: `tasks/tool_router`

Given a user request, the router picks the agent's next action:

- `answer_directly`
- `web_search`
- `read_file`
- `run_code`
- `ask_user`

`gold.jsonl` ships with 22 **synthetic seed rows** so you can test the plumbing. The rules were written with those seeds in view, so the rules score on seeds is inflated. Replace the seeds with 200 real agent inputs:

```bash
python -m zsbench label tasks/tool_router path/to/raw_inputs.jsonl
```

- Inputs are a `.txt` file (one first message per line) or a `.jsonl` file with `text` and an optional `context`.
- Type `2` for one label, or `2,3` for a genuinely two-way row.
- Labeling is resumable.
- Delete the `"source": "seed"` rows once you have real ones.

To check label balance, two-way share, near-duplicates and progress toward 200:

```bash
python -m zsbench check tasks/tool_router
```

### What a gold row looks like

```json
{"id": "a3f9c1e2b7d0", "text": "yes, do it", "context": ["user: the tests are failing again", "assistant: Want me to rerun them with -x?"], "label": "run_code", "split": "test", "source": "real", "note": "alone it reads as ask_user"}
```

| field | required | meaning |
|---|---|---|
| `id`, `text`, `label` | yes | The input and its correct label. |
| `label` as a list | no | `["read_file", "web_search"]` means either label is correct. Keep these under 10% of rows; `check` warns above that. |
| `context` | no | Earlier turns, oldest first. Without it, "yes do it" can't be labeled. Every backend except `rules` sees it. |
| `split` | no | `dev` or `test`. Run with `--split`. |
| `source` | no | `seed` or `real`. |
| `note` | no | Why the row got that label. |

Rules for the set:

- **Use real traffic, messy wording included.** Keep the hard rows. If you drop every row you hesitate on, the set gets easier than reality and the scores flatter you.
- **You make the final call on every row.** A model pre-fill that you confirm is fine, as long as it's an open model. For the trickiest rows, label blind first, because pre-fills pull you toward agreeing.
- **Split dev/test, tune on dev only, report test.** `python -m zsbench split tasks/tool_router` tags untagged rows, about a third of each label to dev.
  - Contrastive pairs and near-copies stay together. Existing tags never move, so new rows just get tagged when you rerun it.
  - When gold has a test split, `run` reports test in the headline and per-label F1.
  - Confusion matrices and the list of misses come from dev only. Individual test errors are never shown, so test stays honest.
  - The accuracy column has a 95% CI. A gap between backends smaller than that CI is noise.
  - Per-label F1 under about 30 rows is a rough guide.
- **Keep gold out of training data.** Before training on model-labeled data, run `python -m zsbench check tasks/tool_router --against train.jsonl`. It flags exact and near-copy leaks (case, punctuation, small rewordings) and exits 1 if it finds any.

To use a different decision, copy the folder and edit the question, the labels (with one-line descriptions) and the rules in `task.json`.

### Hard candidates from models, blind-checked

Real traffic is the best gold, but models can write the hard cases you're short on. Mix generators so the classifier learns the routing signal rather than one model's phrasing:

- Haiku 5.5 for bulk, clear-cut rows.
- Sonnet 5.5 for edge cases.
- DeepSeek (DeepInfra) for a third voice.

Each batch also rotates personas, topics and lengths. Claude models need the project venv (`anthropic` SDK, see `requirements.txt`) and `ANTHROPIC_API_KEY` in `.env`. The checkers come from `.env` (`CHECK_MODELS`, `TIEBREAK_MODEL`) and run on DeepInfra.

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

```bash
.venv/Scripts/python.exe -m zsbench generate tasks/tool_router --model claude-haiku-5-5 --kinds clear,messy --n 100
.venv/Scripts/python.exe -m zsbench generate tasks/tool_router --model claude-sonnet-5-5 --kinds near_miss,needs_context,vague,contrastive_pair,multi_step --n 60
.venv/Scripts/python.exe -m zsbench generate tasks/tool_router --n 40
.venv/Scripts/python.exe -m zsbench blindcheck tasks/tool_router
.venv/Scripts/python.exe -m zsbench label tasks/tool_router tasks/tool_router/to_label.jsonl --source synthetic
.venv/Scripts/python.exe -m zsbench adjudicate tasks/tool_router
```

- With no `--model`, `generate` uses `GEN_MODEL` (DeepSeek).
- `--kinds` cycles through the listed kinds so small runs still cover each one.
- Claude runs use effort `low` for Haiku and `medium` otherwise (override with `--effort`).
- Sonnet requests opt into server-side refusal fallback.

1. **`generate`:** the generator writes candidates, each with an `intended` label, into `candidates.jsonl`. The kinds are clear-cut, near-misses, context-dependent follow-ups, vague requests, contrastive pairs, multi-step asks and messy wording.
   - Topics rotate across 16 domains.
   - Anything that near-copies gold or earlier candidates is dropped.
2. **`blindcheck`:** gpt-oss-120b and MiMo-V2.6-Flash label each candidate seeing only the input and the label menu, never `intended`. When they split, Llama-4-Scout breaks the tie.
   - Every checker matches `intended`, or the tiebreaker sides with it → `teacher.jsonl`. That's training data for later; never gold.
   - Anything else → `to_label.jsonl`. That includes both checkers agreeing with each other against the generator, and invalid answers.
   - Both halves of a contrastive pair always land in the same file, so near-copies never straddle training data and gold.
3. **`label`:** you hand-label the disputed rows into gold, blind: the label tool never shows the votes. `--source synthetic` keeps them counted apart from real traffic. Type `u` to undo your last label.
4. **`adjudicate`** (optional): once you've labeled enough, Sonnet 5.5 settles the rest of the queue.
   - It picks only among the labels the models proposed, and its rows go to `teacher.jsonl`, never gold.
   - Rows you already labeled leave the queue.
   - Pair partners of rows you labeled stay in the queue for you, so pair halves never split between gold and training data.
   - The old queue, with its votes, is archived in `tasks/tool_router/archive/`.

Both commands resume where they left off. `teacher.jsonl` rows record each model's vote and whether agreement was unanimous or tiebreak, so you can filter stricter later.

## Model servers: local GPU locked out

**[2026-10-22] No local model servers on this machine's GPU.** It runs production work. That covers Simple Jev, vLLM, Kev and Intern-Decision.

- The router is DeepSeek V4.1 Flash via API.
- The comparison that needed local models is finished; results are in [docs/jev-landscape.md](docs/jev-landscape.md).
- **The launch commands were removed from this README on purpose.** They're in git history (commit `1b3edb3`) for use on a different machine.
- `scripts/gpu_guard.py launch` refuses to start anything (exit 4), and so does `scripts/intern_server.py`. `gpu_guard.py check` still reports VRAM.
- Lifting the lockout is a deliberate code edit (`LOCKOUT` in `scripts/gpu_guard.py`), on a GPU that doesn't serve PROD. It's not a flag.
- A System One server running somewhere else can still be scored: `--jev-server NAME=URL[,MODEL]` (or `--jev-url`) points the harness at it.

### BFCL routing set (second test set)

`tasks/tool_router_bfcl/` holds 224 of the 250 cases from [pyuvraj/bfcl-v1-jev-routing](https://huggingface.co/datasets/pyuvraj/bfcl-v1-jev-routing) (Apache-2.0), mapped onto our labels by `scripts/map_bfcl.py`. The mapping uses fixed rules, not hand checks:

- lookup tools → `web_search`
- calculation tools → `run_code`
- "no tool fits" rows with numbers → `run_code`; without numbers → `answer_directly`
- bookings, database edits and generators are dropped

It's noisy (for example, "kings of France in the 18th century" maps to `web_search`), and it has no `read_file` or `ask_user` rows. Treat it as a second, rougher signal next to our own test split.

### DeepInfra instead of vLLM

The `llm` backend can call DeepInfra's OpenAI-compatible endpoint instead of a local vLLM server. Its settings live in `.env` at the repo root. `.gitignore` keeps that file out of git, and the key never lands in run files.

1. Open `.env` and paste your key after `LLM_API_KEY=`. It's already set to `deepseek-ai/DeepSeek-V4.1-Flash-1017` on DeepInfra.
2. Run:

   ```bash
   python -m zsbench run tasks/tool_router --backends llm --limit 3
   ```

How the settings resolve:

- `python -m zsbench` reads `.env` on startup.
- Real environment variables win over `.env`, and CLI flags win over both.
- The keys are `LLM_API_KEY`, `LLM_URL`, `LLM_MODEL` and `LLM_EXTRA`.

## Run

The `llm` backend calls DeepSeek on DeepInfra (settings in `.env`, see above). No local server is involved:

```bash
python -m zsbench run tasks/tool_router --backends rules,llm --out runs/<name>
```

To add a remote System One server's results to the same folder:

```bash
python -m zsbench run tasks/tool_router --backends remote --jev-server remote=https://<host> --out runs/<name>
```

Each run prints the comparison table and writes `runs/<name>/report.md`, which includes per-label F1 and confusion matrices. A backend whose server is down is skipped and the run exits 1.

Useful flags:

| flag | what it does |
|---|---|
| `--limit 20` | quick smoke test on the first 20 rows |
| `--split test` | only rows marked `"split": "test"` |
| `--orders 1` | turn off Jev's position-bias averaging |
| `--jev-model`, `--llm-model` | must match what each server is serving |
| `--llm-extra '{}'` | drop the default `enable_thinking: false` (for servers that reject it) |
| `--llm-key-env NAME` | env var holding the `llm` API key (default `LLM_API_KEY`); unset means no auth |

## Reading the report

| column | meaning |
|---|---|
| accuracy | The debiased prediction, with a 95% bootstrap CI. Jev asks once per rotation of the label list, so every label sits first once, and the probabilities are averaged. A two-way row counts as correct if either label is picked. |
| acc, fixed order | Jev asked once with labels in `task.json` order, which is what a naive single call gives you. |
| order flips | Share of inputs where Jev's answer changed with label order. High means strong position bias. |
| mean conf, ECE | How far stated confidence is from real accuracy. Check this before thresholding on confidence. |
| invalid | No usable label: a parse failure or a server error. Error details are in `<backend>.jsonl`. |

## Label wording changelog (`tasks/tool_router/task.json`)

The question and label descriptions are part of every model's prompt, so changing them changes every score. They're versioned, and a test (`WordingFreezeTest`) fails if they change without a new entry here.

### v1, 2026-10-22: frozen

Fingerprint `770dd97d3a97806e`: the first 16 hex characters of the sha256 of `{"question", "labels"}`, serialized with sorted keys.

Rules this wording settles:

- **read_file vs run_code:** reading a value, quoting or summarizing is `read_file`. Any math with specific numbers is `run_code`, even mental arithmetic, and so is executing, measuring or changing a file the user points to.
- **answer_directly vs web_search:** stable knowledge that's still true in a year is `answer_directly`. Facts that change or must be looked up are `web_search`: prices, versions, company practices, salary data, "latest/now".
- **answer_directly vs ask_user:** `ask_user` only when the agent can't act without information only the user has. If a useful answer is possible with a stated assumption, it's `answer_directly`.

How v1 was reached: four drafts on 2026-10-22, each checked against blind-checker agreement or the gold set. The last change was scored on dev only: 14/15 → 15/15.

**No more description edits.** The test split is only about 27 rows, so one row moves the score about 4 points and the 95% interval spans about 30. Wording changes can't be measured reliably at this size. Grow the gold set with real rows instead. A v2 needs a much larger test split to measure it.

## Tests

```bash
.venv/Scripts/python.exe -m unittest discover -s tests -t .
```

The tests use in-process fake servers shaped like the Simple Jev and OpenAI-compatible APIs. They check the plumbing, not model accuracy.
