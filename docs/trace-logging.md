# Trace logging spec (v3)

[2026-10-22] What your agent harness writes so its runs can become a gold set and, later, training data. v3 replaces v2 with the same ideas and far fewer fields.

## Three files

```
traces/<harness>/
  tools.json    the tool menu, once
  steps.jsonl   one line per decision, written before the tool runs
  runs.jsonl    one line per run, written when the run ends
```

### `tools.json`

Each tool's id and one-line description:

```json
{"read_file": "Read a local file or log.",
 "run_code":  "Run a command or script.",
 "web_search": "Look up current or external information.",
 "ask_user":  "Ask the user for missing information.",
 "answer":    "Reply to the user; no tool needed."}
```

### `steps.jsonl`

One line per decision:

```json
{"run": "r42", "step": 4, "parent": 3, "compacted": false,
 "context": [
   {"role": "user", "content": "the backup didn't run again last night"},
   {"role": "tool", "name": "read_file", "content": "rsync: connection refused (111)…"}
 ],
 "tools": ["read_file", "run_code", "ask_user", "answer"],
 "chose": "run_code",
 "router": "run_code"}
```

| field | meaning |
|---|---|
| `run` | The run id |
| `step` | Step number, unique within the run |
| `parent` | The step this one continues, or the step that spawned the sub-agent. `null` for the first step. |
| `compacted` | `true` if the context was summarized right before this step |
| `context` | **Exactly the messages the model saw.** The last one is what it's reacting to. Roles: `user`, `assistant`, `tool` (with `name`), `summary`. |
| `tools` | The tool ids it could pick at this step |
| `chose` | The id it picked |
| `router` | Optional: the router's pick in shadow mode, or `null` |

### `runs.jsonl`

One line per run:

```json
{"run": "r42", "task": "Fix the nightly backup", "model": "claude-sonnet-5-5",
 "outcome": "success", "used": [1, 3, 4], "wrong": [2]}
```

| field | meaning |
|---|---|
| `outcome` | `success`, `fail`, `timeout`, or `error_not_model` (a tool crash or infrastructure failure) |
| `score` | Optional, 0–1, for partial success |
| `used` | Steps whose result was used later. Leave it out if the harness can't tell. |
| `wrong` | Steps you know were bad choices |

## Three rules

1. **`context` is what the model saw at that moment.** After a compaction, that's the summary, not the old turns. Never fill it in with hindsight.
2. **Write the step line before the tool runs.** Write the run line after the run ends.
3. **No secrets.** Redact keys and personal data. Check with `rg -n "sk-|gho_|hf_|Bearer " traces/` before sharing.

## How lines become labels

| run outcome | step becomes |
|---|---|
| `success`, and the step is in `used` | training data: `chose` is the label |
| `success`, but the step isn't in `used` | blind check: winning runs still waste steps |
| `fail` / `timeout` | blind check; `chose` is never trusted as a label |
| `error_not_model` | dropped from everything |
| any step in `wrong` | a hard gold candidate, labeled by hand |

The gold set is always labeled by hand with `zsbench label`.

## Shadow mode

At each step, ask the router too, and put its pick in `router` without acting on it. It's an API call, so the GPU stays free:

```python
from zsbench.router import Router

router = Router()
pick = router.route(last_message, earlier_messages).label
```

## Before handing over

- [ ] Every `chose` is in that step's `tools`, and every id in `tools` is in `tools.json`.
- [ ] Every `run` in `steps.jsonl` has exactly one line in `runs.jsonl`.
- [ ] About 50 steps or more, from about 10 or more tasks, not counting `error_not_model` runs.

## For the converter (not your job)

- **Row text** is the last item of `context`; the earlier items are the row's context.
- **Options** come from `tools` plus `tools.json`.
- **Splits** are grouped by `run`.
- **A row only ever uses its own `context`.** `parent` is for grouping, never for stitching steps together.
- **Fine or coarse labels:** routing over your real tool ids needs per-row options in zsbench. Mapping them onto the 5 router actions works with today's code.
