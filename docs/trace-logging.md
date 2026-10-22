# Trace logging spec: what your harness should write

[2026-10-22] Spec v2. v2 adds branching and compaction, the per-step "result used later" flag, and run outcomes.

This is what an agent harness logs so its runs can become:

- a **gold set**: about 50 decisions you label by hand, used to score the router
- **training data**: thousands of decisions, labeled from successful runs plus the teacher/blind-check pipeline

The format follows the agent-trace datasets that worked for others (AgentDecisionBench: TASK / HISTORY / CURRENT OBSERVATION, with the tool menu as options). A converter into zsbench rows is still to be written; it will read exactly this format.

## The one rule

**Log what the model actually saw at the moment it decided, before it acted.** Not a cleaned-up version, not a later summary, and not earlier steps stitched together. If you shorten anything, mark it (`"truncated": true`). A decision logged with hindsight teaches the router things it will never know in production.

## Runs are trees, not lines

The harness compacts context and may hand work to sub-agents, so one run is a tree of decisions:

- Every decision has a `step_id`, unique within its run.
- **`parent_step_id`** is the step whose context this one continues: the previous step in the same agent, or the step that spawned a sub-agent. It's `null` for the first step.
- **`compacted_before: true`** means the context was rewritten or summarized right before this step. The `history` then holds the summary the model saw, not the turns it replaced.

**Converter rule:** a row's state is built only from that record's own `task`, `history` and `observation`, exactly as logged. It is never built by concatenating earlier steps' histories. That would rebuild context the model didn't see, especially across a compaction. `parent_step_id` is used only to keep a run's branches together (in the same split, for leak checks), never to assemble context.

## Files

- `traces/<harness-name>/<YYYY-MM-DD>.jsonl`, append-only, UTF-8, one JSON object per line.
- There are two record types, told apart by `"type"`:
  - `decision`: one per decision point, written before the action runs
  - `run_end`: one per run, written when the run finishes. Outcome and "was this step's result used later" aren't known until then, so they go here, and the converter joins on `run_id` and `step_id`.

## `decision` record

```json
{
  "type": "decision",
  "spec": 2,
  "run_id": "2026-10-23T19:04:11Z-ab12cd",
  "step_id": "4",
  "parent_step_id": "3",
  "compacted_before": false,
  "ts": "2026-10-23T19:04:15Z",
  "harness": {"name": "home-agent", "version": "0.4.1", "agent_model": "claude-sonnet-5-5"},
  "agent": "main",
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
| `type`, `spec` | yes | `"decision"`, and `2` for this spec version |
| `run_id` | yes | One agent run (the whole tree). Unique across days and machines. |
| `step_id` | yes | Unique within the run. Any string works; e.g. `"4"`, or `"4.sub1.2"` for a sub-agent's step. |
| `parent_step_id` | yes | The step whose context this one continues, or that spawned this sub-agent. `null` for the first step. |
| `compacted_before` | yes | `true` if the context was rewritten or summarized right before this step |
| `ts` | yes | UTC, ISO 8601 |
| `harness` | yes | Name, version, and the model driving the agent |
| `agent` | no | Which agent decided: `main`, or a sub-agent name |
| `task` | yes | The goal this agent is working on, as it was given. A sub-agent logs its own task. |
| `history` | yes | **The context this model saw at this step, oldest first.** After a compaction, that's the summary. Roles: `user`, `assistant`, `tool_call`, `tool_result`, `system`, `summary`. |
| `observation` | yes | The latest thing the agent saw (a user message or tool output), the one it's reacting to |
| `options` | yes | The menu available at this step: each tool's `id` and one-line `description`. Log it every time, even if it never changes; menus drift. |
| `chosen` | yes | The option `id` the agent actually took. It must be one of `options`. |
| `chosen_args` | no | The arguments it called the tool with |
| `decided_by` | yes | Who picked: `agent:<model>`, `router:<model>` or `human` |
| `router` | no | The router's pick if it was consulted, including shadow mode (see below). Copy the `Decision` fields. |

## `run_end` record

```json
{"type": "run_end", "spec": 2, "run_id": "2026-10-23T19:04:11Z-ab12cd", "ts": "2026-10-23T19:06:40Z",
 "outcome": "success", "partial_score": null, "steps": 7, "reason": "cron fixed; backup ran manually",
 "judged_by": "human", "bad_steps": ["2"], "result_used_later": {"1": true, "2": false, "3": true, "4": true}}
```

| field | required | meaning |
|---|---|---|
| `outcome` | yes | One of the four outcomes below |
| `partial_score` | no | 0–1, how much of the task got done, when the harness can score it |
| `steps` | yes | The number of `decision` records in the run, across all branches |
| `reason` | no | One line on why the run ended this way |
| `judged_by` | yes | `human`, `test` (an automatic check), or `agent` (self-report; weakest) |
| `bad_steps` | no | The `step_id`s you know were wrong. These are the most valuable label you can give. |
| `result_used_later` | no | `{step_id: true or false}`: did any later step, or the final answer, reference this step's result? List only the steps the harness can tell about; any missing step counts as `null` (unknown). |

The four outcomes:

| outcome | meaning |
|---|---|
| `success` | The run achieved the task |
| `fail` | The run ended without achieving the task |
| `timeout` | The run hit a step or time limit |
| `error_not_model` | A tool crash or infrastructure failure, not the model's doing |

## How runs become labels

| run outcome | its steps become |
|---|---|
| `success` | Teacher labels (`label = chosen`) **only** for steps with `result_used_later: true` that aren't in `bad_steps`. Winning runs still contain wasted steps, so steps with `false` or `null` go to blind checking instead. |
| `fail` / `timeout` | Blind checking. `chosen` is never used as a teacher label. `bad_steps` are recorded as known-wrong. |
| `error_not_model` | **Excluded from training and from gold entirely.** The failure says nothing about the decisions. |
| any outcome | `bad_steps` are known-wrong answers. They make good hard gold candidates, labeled by hand. |

Gold is always labeled by hand with `zsbench label`, whatever the outcome.

## Shadow mode: free agreement data

At each decision, let the agent act as it normally would, but also ask the router and log its pick in `router`:

```python
from zsbench.router import Router

router = Router()
d = router.route(observation, history_as_turns)
```

This changes nothing in the agent's behavior, and every run then records the agent's choice and the router's choice side by side. Where they disagree is exactly where to look first. The router is an API call, so shadow mode puts no load on the local GPU.

## Privacy and safety

- **Never log secrets.** Redact API keys, tokens, passwords and `.env` values before writing. Search before sharing: `rg -n "sk-|gho_|hf_|Bearer " traces/`.
- Strip personal data that the decision doesn't need (emails, phone numbers, addresses).
- Cap very long tool results at about 4,000 characters, keeping the start and end, and set `"truncated": true`.

## How the converter will use it

| zsbench field | comes from |
|---|---|
| `text` | `observation` |
| `context` | This record's `task`, then its `history` rendered as `role: content` lines (tool calls as `tool_call: name(args)`). Nothing from other records. |
| `label` (gold) | your hand label from `zsbench label` on sampled decisions |
| `label` (teacher) | see "How runs become labels" |
| options | `options` |
| group (splits and leak checks) | `run_id`: all steps and branches of a run stay in the same split |

Two choices are left for when real traces exist:

1. **Coarse or fine labels.** We can map your harness's tools onto the 5 router actions (cheap; works with today's code), or route over the real tool ids per step. The fine option needs per-row options in zsbench, since today it uses one fixed menu per task.
2. **Gold sampling.** Stratify by `chosen` and by `outcome`, and over-sample steps where the agent and router disagree.

## Checklist before handing traces over

- [ ] Every `decision` has `options`, and `chosen` is one of their ids.
- [ ] `step_id`s are unique within each run, and every `parent_step_id` is `null` or an existing `step_id` in the same run.
- [ ] Every step after a compaction has `compacted_before: true`, and its `history` is the summary the model saw.
- [ ] No `history` contains anything from later steps or from sibling branches.
- [ ] Every `run_id` has exactly one `run_end`, with `outcome` set to one of the four values.
- [ ] The secret search above finds nothing.
- [ ] At least about 50 decisions, from at least about 10 different tasks, excluding `error_not_model` runs.
