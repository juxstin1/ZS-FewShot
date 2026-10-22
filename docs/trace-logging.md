# Trace logging spec

**Spec v2**, written 2026-10-22. This is what your agent harness writes so its runs can become:

- a **gold set**: about 50 decisions labeled by hand, used to score the router
- **training data**: labeled from successful runs plus the blind-check pipeline

---

## At a glance

```
traces/<harness-name>/<YYYY-MM-DD>.jsonl     append-only · UTF-8 · one JSON object per line

  {"type": "decision", ...}    one per decision     written BEFORE the tool runs
  {"type": "decision", ...}
  {"type": "run_end",  ...}    one per run          written when the run finishes
```

| when | write | key fields |
|---|---|---|
| The agent is about to call a tool | `decision` | what it saw (`history`, `observation`), what it could pick (`options`), what it picked (`chosen`) |
| The run is over | `run_end` | `outcome`, `bad_steps`, `result_used_later` |

---

## Schema

Written as TypeScript-style types for readability. `?` means optional.

### `decision`: one per decision point

```ts
interface Decision {
  type:             "decision";
  spec:             2;

  // where this step sits in the run
  run_id:           string;          // one run (the whole tree), unique across machines and days
  step_id:          string;          // unique within the run: "4", or "4.sub1.2" for a sub-agent's step
  parent_step_id:   string | null;   // the step this continues, or that spawned this sub-agent; null for the first step
  compacted_before: boolean;         // the context was summarized right before this step
  ts:               string;          // UTC, ISO 8601

  // who is deciding
  harness:          { name: string; version: string; agent_model: string };
  agent?:           string;          // "main", or a sub-agent name
  decided_by:       string;          // "agent:<model>" | "router:<model>" | "human"

  // what it saw: EXACTLY the context the model had at this step
  task:             string;          // the goal this agent was given
  history:          Message[];       // oldest first; after a compaction, the summary
  observation:      string;          // the latest input, the thing it's reacting to

  // what it could do, and what it did
  options:          Option[];        // the menu at THIS step; log it every time
  chosen:           string;          // must be one of options[].id
  chosen_args?:     object;          // the arguments it called the tool with

  // shadow mode (optional)
  router?:          RouterPick;      // the router's pick, logged but not acted on
}

interface Message {
  role:        "user" | "assistant" | "tool_call" | "tool_result" | "system" | "summary";
  content?:    string;
  tool?:       string;               // for tool_call / tool_result
  args?:       object;               // for tool_call
  error?:      boolean;              // for tool_result
  truncated?:  boolean;              // true if you shortened it
}

interface Option     { id: string; description: string }
interface RouterPick { label: string; ok: boolean; latency_ms: number; model: string; wording_version: string }
```

### `run_end`: one per run

```ts
interface RunEnd {
  type:               "run_end";
  spec:               2;
  run_id:             string;
  ts:                 string;

  outcome:            Outcome;
  partial_score?:     number;                   // 0 to 1, if the harness can grade partial success
  steps:              number;                   // the number of decision lines in the run, across all branches
  reason?:            string;                   // one line on why it ended this way
  judged_by:          "human" | "test" | "agent";   // who decided the outcome ("agent" is weakest)

  bad_steps?:         string[];                 // step_ids you know were wrong choices; the most valuable label
  result_used_later?: Record<string, boolean>;  // step_id -> was its result used by a later step or the final answer?
}                                               //   steps left out count as unknown

type Outcome =
  | "success"           // the task was achieved
  | "fail"              // ended without achieving it
  | "timeout"           // hit a step or time limit
  | "error_not_model";  // a tool crash or infrastructure failure, not the model's doing
```

---

## Examples

A `decision` line (pretty-printed here; on disk it's one line):

```json
{
  "type": "decision",
  "spec": 2,

  "run_id": "2026-10-23T19:04:11Z-ab12cd",
  "step_id": "4",
  "parent_step_id": "3",
  "compacted_before": false,
  "ts": "2026-10-23T19:04:15Z",

  "harness": { "name": "home-agent", "version": "0.4.1", "agent_model": "claude-sonnet-5-5" },
  "agent": "main",
  "decided_by": "agent:claude-sonnet-5-5",

  "task": "Find why the nightly backup failed and fix the cron entry.",
  "history": [
    { "role": "user",        "content": "the backup didn't run again last night" },
    { "role": "assistant",   "content": "Checking the backup log first." },
    { "role": "tool_call",   "tool": "read_file", "args": { "path": "/var/log/backup.log" } },
    { "role": "tool_result", "tool": "read_file", "content": "rsync: connection refused (111)…", "error": false, "truncated": true }
  ],
  "observation": "rsync: connection refused (111) at 02:00:03",

  "options": [
    { "id": "answer_directly", "description": "Reply to the user now; no tool needed." },
    { "id": "web_search",      "description": "Look up current or external information." },
    { "id": "read_file",       "description": "Read a local file, log or document." },
    { "id": "run_code",        "description": "Run a command, script or calculation." },
    { "id": "ask_user",        "description": "Ask the user for missing information before acting." }
  ],
  "chosen": "run_code",
  "chosen_args": { "command": "systemctl status rsyncd" },

  "router": { "label": "run_code", "ok": true, "latency_ms": 640, "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017", "wording_version": "v1" }
}
```

A `run_end` line:

```json
{
  "type": "run_end",
  "spec": 2,
  "run_id": "2026-10-23T19:04:11Z-ab12cd",
  "ts": "2026-10-23T19:06:40Z",

  "outcome": "success",
  "partial_score": null,
  "steps": 7,
  "reason": "cron fixed; backup ran manually",
  "judged_by": "human",

  "bad_steps": ["2"],
  "result_used_later": { "1": true, "2": false, "3": true, "4": true }
}
```

---

## Rules

1. **Log what the model saw, at the moment it decided.**
   - `history` is the context it actually had. After a compaction, that means the summary, not the old turns.
   - Never fill anything in with hindsight. Mark anything you shorten with `"truncated": true`.
2. **Runs are trees.** Sub-agents and compactions branch a run. Use `parent_step_id` to link steps; don't merge them into one history.
3. **Write `decision` before the tool runs**, and `run_end` after the run ends. Outcome and "used later" aren't known before then.
4. **Log `options` on every step,** even if they never change. Menus drift.
5. **No secrets.**
   - Redact keys, tokens, passwords and personal data.
   - Cap long tool output at about 4,000 characters, keeping the start and end.
   - Check before sharing: `rg -n "sk-|gho_|hf_|Bearer " traces/`.

---

## How runs become labels

| outcome | a step becomes |
|---|---|
| `success`, and `result_used_later` is `true` | **training data**: `chosen` is the label |
| `success`, and `result_used_later` is `false` or unknown | **blind check**: winning runs still waste steps |
| `fail` or `timeout` | **blind check**: `chosen` is never trusted |
| `error_not_model` | **dropped** from training and gold |
| listed in `bad_steps`, any outcome | **hard gold candidate**, labeled by hand |

The gold set is always labeled by hand, with `zsbench label`.

---

## Shadow mode

Ask the router at every step, log its pick in `router`, and don't act on it. Disagreements between the router and your agent are where to look first. It's an API call, so the GPU stays untouched.

```python
from zsbench.router import Router

router = Router()
decision["router"] = router.route(observation, history_as_text_turns).to_dict()
```

---

## Before handing traces over

- [ ] `chosen` is one of `options[].id` on every `decision`.
- [ ] `step_id`s are unique within each run, and every `parent_step_id` is `null` or an existing `step_id` in the same run.
- [ ] Every step right after a compaction has `compacted_before: true`, and its `history` is the summary the model saw.
- [ ] No `history` contains anything from later steps or from sibling branches.
- [ ] Every `run_id` has exactly one `run_end`, with a valid `outcome`.
- [ ] The secret search above finds nothing.
- [ ] About 50 decisions or more, from about 10 or more tasks, not counting `error_not_model` runs.

---

## For the converter (not your job)

| zsbench row | comes from |
|---|---|
| `text` | `observation` |
| `context` | this record's `task` and `history` only, never other records |
| options | `options` |
| gold `label` | hand label from `zsbench label` |
| teacher `label` | the "How runs become labels" table above |
| split group | `run_id`: a whole run stays in one split |

Still open: map your tools onto the 5 router actions (works with today's code), or route over your real tool ids (needs per-row options in zsbench).
