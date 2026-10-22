# Trace logging spec: what your harness should write

[2026-10-22] Spec v1. This is what an agent harness logs so its runs can become:

- a **gold set**: about 50 decisions you label by hand, used to score the router
- **training data**: thousands of decisions, labeled from successful runs plus the teacher/blind-check pipeline

The format follows the agent-trace datasets that worked for others (AgentDecisionBench: TASK / HISTORY / CURRENT OBSERVATION, with the tool menu as options). A converter into zsbench rows is still to be written; it will read exactly this format.

## The one rule

**Log what the agent actually saw at the moment it decided, before it acted.** Not a cleaned-up version and not a later summary. If you shorten anything, mark it (`"truncated": true`). A decision logged with hindsight teaches the router things it will never know in production.

## Files

- `traces/<harness-name>/<YYYY-MM-DD>.jsonl`, append-only, UTF-8, one JSON object per line.
- There are two record types, told apart by `"type"`:
  - `decision`: one per decision point, written before the action runs
  - `run_end`: one per run, written when the episode finishes. Outcome isn't known until then, so it gets its own line, and the converter joins on `run_id`.

## `decision` record

```json
{
  "type": "decision",
  "spec": 1,
  "run_id": "2026-10-23T19:04:11Z-ab12cd",
  "step": 3,
  "ts": "2026-10-23T19:04:15Z",
  "harness": {"name": "home-agent", "version": "0.4.1", "agent_model": "claude-sonnet-5-5"},
  "task": "Find why the nightly backup failed and fix the cron entry.",
  "history": [
    {"role": "user", "content": "the backup didn't run again last night"},
    {"role": "assistant", "content": "Checking the backup log first."},
    {"role": "tool_call", "tool": "read_file", "args": {"path": "/var/log/backup.log"}},
    {"role": "tool_result", "tool": "read_file", "content": "rsync: connection refused (111)…", "error": false, "truncated": true}
  ],
  "observation": "rsync: connection refused (111) at 02:00:03",
  "options": [
    {"id": "answer_directly", "description": "Reply to the user now; no tool needed."},
    {"id": "web_search", "description": "Look up current or external information."},
    {"id": "read_file", "description": "Read a local file, log or document."},
    {"id": "run_code", "description": "Run a command, script or calculation."},
    {"id": "ask_user", "description": "Ask the user for missing information before acting."}
  ],
  "chosen": "run_code",
  "chosen_args": {"command": "systemctl status rsyncd"},
  "decided_by": "agent:claude-sonnet-5-5",
  "router": {"label": "run_code", "ok": true, "latency_ms": 640, "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017", "wording_version": "v1"}
}
```

| field | required | meaning |
|---|---|---|
| `type`, `spec` | yes | `"decision"`, and `1` for this spec version |
| `run_id` | yes | One agent episode. Unique across days and machines. |
| `step` | yes | 0-based index of this decision within the run |
| `ts` | yes | UTC, ISO 8601 |
| `harness` | yes | Name, version, and the model driving the agent |
| `task` | yes | The goal the agent is working on, as it was given. Use the first user message if there's no separate task. |
| `history` | yes | Everything before this decision, oldest first. Roles: `user`, `assistant`, `tool_call`, `tool_result`, `system`. Empty list at step 0. |
| `observation` | yes | The latest thing the agent saw (a user message or tool output), the one it's reacting to |
| `options` | yes | The menu available at this step: each tool's `id` and one-line `description`. Log it every time, even if it never changes; menus drift. |
| `chosen` | yes | The option `id` the agent actually took. It must be one of `options`. |
| `chosen_args` | no | The arguments it called the tool with |
| `decided_by` | yes | Who picked: `agent:<model>`, `router:<model>` or `human` |
| `router` | no | The router's pick if it was consulted, including shadow mode (see below). Copy the `Decision` fields. |

## `run_end` record

```json
{"type": "run_end", "spec": 1, "run_id": "2026-10-23T19:04:11Z-ab12cd", "ts": "2026-10-23T19:06:40Z",
 "success": true, "steps": 6, "reason": "cron fixed; backup ran manually", "judged_by": "human"}
```

| field | required | meaning |
|---|---|---|
| `success` | yes | Did the run achieve the task: `true`, `false`, or `null` if unknown |
| `steps` | yes | The number of `decision` records in the run |
| `reason` | no | One line on why it succeeded or failed |
| `judged_by` | yes | `human`, `test` (an automatic check), or `agent` (self-report; weakest) |
| `bad_steps` | no | Step numbers you know were wrong, e.g. `[2]`. These are the most valuable label you can give. |

## Shadow mode: free agreement data

At each decision, let the agent act as it normally would, but also ask the router and log its pick in `router`:

```python
from zsbench.router import Router

router = Router()
d = router.route(observation, history_as_turns)
```

This changes nothing in the agent's behavior, and every run then records the agent's choice and the router's choice side by side. Where they disagree is exactly where to look first.

## Privacy and safety

- **Never log secrets.** Redact API keys, tokens, passwords and `.env` values before writing. Search before sharing: `rg -n "sk-|gho_|hf_|Bearer " traces/`.
- Strip personal data that the decision doesn't need (emails, phone numbers, addresses).
- Cap very long tool results at about 4,000 characters, keeping the start and end, and set `"truncated": true`.

## How the converter will use it

| zsbench field | comes from |
|---|---|
| `text` | `observation` |
| `context` | `task`, then `history` rendered as `role: content` lines (tool calls as `tool_call: name(args)`) |
| `label` (gold) | your hand label from `zsbench label` on sampled decisions |
| `label` (teacher) | `chosen` from runs with `success: true` and no `bad_steps`, plus the blind-check / adjudicate pipeline on the rest |
| options | `options` |

Two choices are left for when real traces exist:

1. **Coarse or fine labels.** We can map your harness's tools onto the 5 router actions (cheap; works with today's code), or route over the real tool ids per step. The fine option needs per-row options in zsbench, since today it uses one fixed menu per task.
2. **Gold sampling.** Stratify by `chosen` and by `success`, and over-sample steps where the agent and router disagree.

## Checklist before handing traces over

- [ ] Every `decision` has `options`, and `chosen` is one of their ids.
- [ ] Every `run_id` has exactly one `run_end`.
- [ ] `history` at step N contains nothing that happened after step N.
- [ ] The secret search above finds nothing.
- [ ] At least about 50 decisions, from at least about 10 different tasks.
