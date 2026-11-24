# Free Jev-style classifiers: where their data came from, and what it looks like

[2026-10-22] Pulled from Hugging Face model and dataset cards, the datasets-server row API, and the GitHub READMEs linked below. Download counts were current on that date. Benchmark numbers are each project's own and unverified.

## The short version

Nobody publishes real user traffic. Every free Jev-style model got its data one of four ways:

| how | who | labels come from | license fit |
|---|---|---|---|
| **1. Convert public labeled datasets** into the Jev shape (state + typed question + options) | Kev-0.5B, tasksource-jev (5.5M rows), Laurer zero-shot | the original human annotations | depends on each source dataset |
| **2. Synthetic "controlled" generators with code-computed labels** | Open-Jev (CC0, 520k rows), Kev's rule/policy records | deterministic rules, with fractional targets where a case is truly ambiguous | clean |
| **3. Distill Jev itself** (send states to Jev and keep its probability distributions) | jev-distill-corpus-v3 (741k rows), which trains autotrust JEV-9B/27B/35B, the most-downloaded open "JEV" models | Jev 1.13 via OpenRouter | TypeSafe's terms apply |
| **4. Run a simulated agent environment and record its states** | AgentDecisionBench / DecisionGym (paper arXiv 2610.21412) | a canonical plan per task lists the acceptable tools at each step | CC-BY (bench) / CC-BY-NC (train) |

**Number 4 is our use case.** It routes an agent's next step, and it shows the biggest measured gain from training: AgentDecision-4B scored 94.9% on unseen states vs 52.9% for untrained Qwen3-4B (the authors' numbers).

## What their traces actually look like

These are real rows from the datasets, trimmed.

**Agent next-step decision** ([JevAgentHarness/AgentDecision-train](https://huggingface.co/datasets/JevAgentHarness/AgentDecision-train), 23,777 rows):

```text
TASK: Refund the full amount of Jia Haddad's most recent delivered order (highest order id) as store credit.

HISTORY:
[0] find_user({"name": "Jia Haddad"}) -> {"user_id": "U5729", "name": "Jia Haddad", "tier": "silver"}
[1] get_orders({"user_id": "U5729", "status": "delivered"}) -> [{"order_id": "ORD-2840", ... "amount": 173.57}]
[2] refund({"order_id": "ORD-2840", "amount": 173.57, "method": "store_credit"}) -> Refunded 173.57 via store_credit.

CURRENT OBSERVATION:
Refunded 173.57 via store_credit.
```

- `question`: "Which tool should the agent call next?"
- `options`: about 60 tools, each written as "name -- description"; many are near-miss distractors.
- `target`: a soft distribution (Jev's) plus gold.
- Other fields: `dp` (decision point: tool_select, stop, clarify …), `after_error`, `episode_success`.
- Noise is injected on purpose: hundreds of `log: heartbeat ok #93621` lines are spliced into the history.

**Tool routing** ([pyuvraj/bfcl-v1-jev-routing](https://huggingface.co/datasets/pyuvraj/bfcl-v1-jev-routing), 250 frozen eval cases from Berkeley's function-calling benchmark, Apache-2.0):

```text
state: Actions this turn: none yet
       User request: Calculate the area of a triangle, given the lengths of its three sides: 3, 4, and 5.
instruction: Which capability should the assistant use for its next step?
options:
  no_tool       - answer the user now: the request needs no tool or is already answered
  clarify       - ask the user which one or what exactly: the request is ambiguous
  cannot_answer - decline: no tool here can send email, messages, payments or do this
  math.triangle_area_heron, ...   (real tools, with distractors)
gold_action_id: math.triangle_area_heron
```

Its menu maps almost one-to-one onto ours:

| theirs | ours |
|---|---|
| no_tool | answer_directly |
| clarify | ask_user |
| tools | web_search / read_file / run_code |
| cannot_answer | (no equivalent) |

**Typed decisions** ([LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions), 3,200 rows, Apache-2.0):

- The state is a JSON object: an agent's task, its constraints, and a trace summary (steps, tool errors, irreversible actions).
- Five typed questions are asked at once: action (continue / observe / human_review / stop), needs_review, risk, urgency and outcome.
- Every gold answer is a probability distribution.

**Controlled synthetic** ([ZefanCai/Open-Jev](https://huggingface.co/datasets/ZefanCai/Open-Jev), 520k rows, CC0):

- Each question spells out its own decision rule, for example: "A malfunction of an existing feature is bug_report; a payment or refund request is billing … Secondary context does not override an explicit priority."
- Where a case is genuinely ambiguous, the target splits evenly, e.g. `[0.5, 0.5]`.

**Distilled** ([SargeDev/jev-distill-corpus-v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3)):

- Short synthetic scenarios: "A statistic in the draft attributes to a study that says the opposite …"
- Question: "Rate claim reliability … 0-5."
- Target: Jev's distribution, `[0.01, 0.01, 0.05, 0.41, 0.51, 0.01]`.

## Free models you can run

| model | size / base | license | trained on | reported result |
|---|---|---|---|---|
| [internlm/Intern-Decision-4B](https://huggingface.co/internlm/Intern-Decision-4B) (also 0.8B, 2B) | Qwen3.5-4B | Apache-2.0 | "decision tuning" (see arXiv 2610.22131) | avg 90.0 vs Jev 88.7 on its table; Jev-style response envelope |
| [jaredpalmer/kev-4b](https://huggingface.co/jaredpalmer/kev-4b) (also 0.8B / 9B / 27B, GGUF) | Qwen3.5 + LoRA + pointer head | Apache-2.0 | public datasets + rule records, open-weight teachers only | ~0.79 out-of-domain vs Jev 0.86 |
| [autotrust/JEV-9B](https://huggingface.co/autotrust/JEV-9B) (also 27B, 35B) | Qwen3.5-9B + LoRA head | Apache-2.0 | distilled from Jev 1.13 | 90% top-1 agreement with Jev, ~90 ms/decision |
| [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) | ModernBERT-large, 421M | Apache-2.0 | not stated | ~33 ms on a T4 |
| [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) (GGUF exists) | Qwen 9B + scoring head | Apache-2.0 | internal synthetic data (private) | needs ~41 GB VRAM unquantized |
| [JevAgentHarness/AgentDecision-4B](https://huggingface.co/JevAgentHarness/AgentDecision-4B) | Qwen3-4B + LoRA | **CC-BY-NC** | DecisionGym agent states + Jev soft labels + public tool-use data | 94.9% on shifted agent states vs 52.9% untrained |

## Measured on our data (2026-10-22)

Runs are in `runs/compare-test/` (our test split, n=27) and `runs/compare-bfcl/` (BFCL mapped by rule, n=224). Wording is v1, and each local model was asked once per rotation of the label order, with the results averaged.

| backend | our test | BFCL mapped | p50 latency | answer changed with label order | accuracy, single fixed order (test / BFCL) | VRAM |
|---|---|---|---|---|---|---|
| rules | 55.6% (37–74) | 46.0% (40–52) | ~0 ms | – | – | – |
| DeepSeek V4.1 Flash (DeepInfra API) | 88.9% (74–100) | **78.6%** (73–84) | 1,380 ms | – | – | none |
| Intern-Decision-4B (local) | 88.9% (74–100) | 75.4% (70–81) | 423 ms | 33% / 35% | 67% / 69% | ~11 GB |
| Kev-4B (local, CUDA graphs off) | 88.9% (74–100) | 70.5% (65–76) | **260 ms** | 19% / 18% | 85% / 68% | ~10 GB |

- **Our test split can't separate the three models.** All got 24 of 27, and the CI spans about 26 points.
- **BFCL separates them a little:** DeepSeek ≥ Intern > Kev. Its labels are rule-mapped, though, and it has no read_file or ask_user rows.
- **Both local models depend heavily on order averaging.** Asked once, Intern drops to 67% on our test because a third of its answers change with label order. Kev is steadier, but its BFCL score still falls from 70.5% to 68% when asked once.
- **The GPU is shared with production work.** Every launch went through `scripts/gpu_guard.py`: preflight, a 12 GB cap, and a 2 GB free-VRAM watchdog, one model at a time.
  - With graphs on, Kev hit the cap and failed with out-of-memory inside its own process, which is the intended behavior.
  - VRAM is protected, but compute isn't: the GPU showed up to 100% busy during our scoring runs.

## What this means for us

1. **Stop making data. Test the free models on our gold set.**
   - Intern-Decision-4B and Kev-4B both fit on the 4090.
   - Score them on our 27-row test split plus the 250-case BFCL routing set, which is a second, independent check of the same task.
2. **Copy their trace format if the router will sit inside a real agent.** That means TASK / HISTORY of tool calls and results / CURRENT OBSERVATION, with tool names and one-line descriptions as options. Our current rows are a user message plus a few turns, so they're closer to the BFCL routing set.
3. **If we ever train, DecisionGym is the recipe to copy:**
   - a scripted environment with a canonical plan per task, so labels come from code, not opinion
   - near-miss distractor tools
   - injected log noise
   - a "shifted" variant (renamed tools, new task kinds) as the out-of-distribution test

## Sources

- [AgentDecisionBench](https://huggingface.co/datasets/JevAgentHarness/AgentDecisionBench) · [AgentDecision-train](https://huggingface.co/datasets/JevAgentHarness/AgentDecision-train) · [JevAgentHarness code](https://github.com/JevAgentHarness/JevAgentHarness)
- [bfcl-v1-jev-routing](https://huggingface.co/datasets/pyuvraj/bfcl-v1-jev-routing)
- [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) · [tasksource-jev-typed-decisions](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions) · [Open-Jev](https://huggingface.co/datasets/ZefanCai/Open-Jev) · [jev-distill-corpus-v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3)
- Models: [Intern-Decision-4B](https://huggingface.co/internlm/Intern-Decision-4B) · [kev-4b](https://huggingface.co/jaredpalmer/kev-4b) · [JEV-9B](https://huggingface.co/autotrust/JEV-9B) · [laya](https://huggingface.co/convaiinnovations/laya) · [clef-flash](https://huggingface.co/Cloudflare/clef-flash) · [AgentDecision-4B](https://huggingface.co/JevAgentHarness/AgentDecision-4B)
