# State

[2026-10-22]

- **Repo:** https://github.com/juxstin1/ZS-FewShot (private), branch `main`.
  - Not in git: `.env` (keys), `.venv/`, and `servers/` (model servers and weights). Rebuild them from the README.
- **Open decision:** what the router plugs into. Options:
  - an existing agent: swap the labels for its real tools, and feed it task + history + observation
  - a small demo agent
  - nothing yet

  Training is only worth it for a small local model, and only on traces from that agent.
- **[2026-10-22] Direction:** Justin will generate traces with his own agent harness at home.
  - Logging spec v2: `docs/trace-logging.md`, rewritten 2026-10-22 as typed schemas with examples.
    - The v3 field cuts were reverted; Justin wanted the doc clearer, not fewer fields.
    - One JSONL file per day, with `decision` lines (run/step tree, compaction flag, exact context, options, chosen, router) and `run_end` lines (outcome, bad_steps, result_used_later).
  - Labels:
    - Successful-run steps whose result was used later become training data.
    - Other steps go to blind checking.
    - `error_not_model` runs are dropped.
  - Next code: a trace-to-zsbench converter, plus per-row options if the tool menu varies per step.
- **[2026-10-22] Free decision models scored** (details and table in `docs/jev-landscape.md`, "Measured on our data"):

  | backend | our test (n=27) | BFCL mapped (n=224) | p50 latency |
  |---|---|---|---|
  | DeepSeek API | 88.9% | 78.6% | 1,380 ms |
  | Intern-Decision-4B | 88.9% | 75.4% | 423 ms |
  | Kev-4B | 88.9% | 70.5% | 260 ms |
  | rules | 55.6% | 46.0% | – |

  - Local models need label-order averaging (asked once, Intern scores 67% on test).
  - **[2026-10-22] Local model servers are LOCKED OUT on this machine** (its GPU serves production work).
    - `gpu_guard.py launch` exits 4, and `intern_server.py` refuses to start.
    - The README launch commands are removed; they're in git history at `1b3edb3`.
    - The `servers/` folder stays on disk but unused.
  - WSL has no network (mirrored mode, no route), so everything runs on Windows. Kev needs `HF_HUB_DISABLE_SYMLINKS=1` and `KEV_CUDA_GRAPHS=0`.
- **BFCL set:** `tasks/tool_router_bfcl/` holds 224 of 250 rows, mapped by `scripts/map_bfcl.py` (rules, not hand-checked).
- **Report:** test-only runs now hide confusion matrices too.
- **[2026-10-22] Router shipped:** DeepSeek V4.1 Flash on DeepInfra.
  - `zsbench/router.py`: `Router().route(text, context)` returns a `Decision` with label, ok, latency_ms, attempts, model and wording_version. It retries once, then falls back to answer_directly.
  - CLI: `python -m zsbench route "..." --context ...`.
  - Live smoke: 5/5 correct, 0.6–2.7 s.
  - **Not yet wired into an agent**: we need the agent's codebase. `load_dotenv` moved to `zsbench/env.py`.
- **Tests:** 59/59 pass (`.venv/Scripts/python.exe -m unittest discover -s tests -t .`). Tests no longer assume the live gold size.
- **[2026-10-22] Gold is split 15 dev / 27 test** via `python -m zsbench split`.
  - Stratified by label, with pairs and near-copies kept together. Existing tags never move.
  - The pre-split copy is in `tasks/tool_router/archive/gold.before-split.jsonl`.
  - **Rule:** tune on dev only, report test. With a test split, `run` reports test in the headline and per-label F1, and shows confusion and misses for dev only.
- **[2026-10-22] Wording v1 is frozen** (fingerprint `770dd97d3a97806e`; `WordingFreezeTest` enforces it; README changelog).
  - The final run_code edit: any math with specific numbers, plus executing, measuring or changing a file the user points to.
  - Scored on dev only, as Justin asked: DeepSeek 14/15 → 15/15. The one change: "days between 2026-03-14 and 2027-01-02" went from answer_directly to run_code. Test was not run.
  - **No more description edits; grow gold with real rows next.**
- **[2026-10-22] Gold now has 42 rows:** 22 seeds and 20 synthetic rows Justin labeled from the disputed queue.
  - Three rows were relabeled after a review pass, each with a `note`: Zigbee "rule of thumb" → answer_directly, "cut the low end, thoughts?" → answer_directly, drum timing → ask_user.
  - Justin agreed with the generator on 12 of his first 17 rows.
- **`adjudicate`** (Sonnet 5.5, effort medium) settled the other 29 disputed rows into teacher.jsonl. It overruled the generator on 6.
  - One row is left in to_label.jsonl: the pair partner of a row Justin labeled.
  - Leak check `check --against teacher.jsonl` is clean. teacher.jsonl has 179 rows.
- **Label CLI** has a `u` key that undoes the last label.
- **[2026-10-22] First baseline on 42-row gold** (`runs/gold42-deepseek/`):

  | backend | overall (95% CI) | seeds | synthetic |
  |---|---|---|---|
  | rules | 57.1% (43–71) | 64% | 50% |
  | DeepSeek V4.1 Flash | 92.9% (86–100), p50 637 ms | 22/22 | 17/20 |

  - All 3 DeepSeek misses are gold run_code predicted as read_file, where the request acts on a file: run a script to check it, measure audio stem loudness, rebalance a mix. The run_code description doesn't yet say "execute, measure or modify files".
  - Caveat: DeepSeek generated some of those synthetic rows.
- **[2026-10-22] v4 run_code wording** ("executes, measures or modifies files"; read_file points to run_code for running, measuring or changing). Re-run in `runs/gold42-deepseek-v4wording/`:
  - DeepSeek is still 92.9%, but the errors moved: fixed 3 file-action rows, broke 3 others.
    - Broken: two arithmetic seeds ("17% of 48,250…", "days between dates") now go to answer_directly.
    - Broken: drum timing (gold ask_user) now goes to run_code.
  - Rules are unchanged at 57.1%.
  - **This wording was tuned while looking at gold**, so scores from here on are optimistic. Split gold into dev/test before more tweaks.
- **[2026-10-22] Hardened edge-case prompts:**
  - A new `boundary` kind targets one (correct, tempting) label pair per batch from `BOUNDARY_PAIRS`. It returns `rival` and `why`, which checkers never see.
  - All edge kinds now forbid giveaway keywords.
  - Sonnet batch of 40 at effort high, vs the previous Sonnet prompts:

    | | previous | hardened |
    |---|---|---|
    | disputed | 10% | 38% |
    | unanimous | 83% | 48% |
    | checkers agree with each other | 92% | 57% |

  - By kind: boundary 75% disputed (only 8 rows), contrastive_pair 50%, near_miss 6% (still too easy).
  - Disputes cluster on date math (run_code vs answer_directly). Whether disputed rows are real boundaries or generator mistakes shows up when Justin labels them.
  - to_label.jsonl now has 50 rows.
- **Env:** `.venv` (Python 3.12) with `anthropic==2.1.0` for the Claude generators (`requirements.txt`). Everything else is stdlib.
- **[2026-10-22] Claude generators are live.** Haiku 5.5 handles bulk (clear, messy); Sonnet 5.5 handles edge cases (near_miss, needs_context, vague, contrastive_pair, multi_step). Batches rotate 8 personas.

  96 new candidates were blind-checked:

  | generator | n | unanimous | to label | checkers agree with each other |
  |---|---|---|---|---|
  | Haiku | 48 | 75% | 8% | 79% |
  | Sonnet | 48 | 83% | 10% | 92% |
  | DeepSeek | 64 | 44% | 41% | 77% |

  DeepSeek's rows predate the v3 wording, so that comparison is skewed. Totals: 125 rows in teacher.jsonl, 35 in to_label.jsonl.
- Sonnet's "edge cases" got high checker agreement, so they may be too clean to be hard. Watch the dispute rate if we want gold candidates from them.
- **Synthetic pipeline** (`zsbench/synth.py`, commands `generate` / `blindcheck`): 64 candidates so far.
  - Sorted into 46 rows in `teacher.jsonl` and 18 for hand-labeling in `tasks/tool_router/to_label.jsonl`.
  - The first sort, made under the old label wording, is in `tasks/tool_router/archive/`.
- **[2026-10-22] Label boundary:** "read a file then compute" is `run_code`. `read_file` means reading is the answer: summarize, explain, quote, or look up values or matching entries.
  - Re-checking batch 1 under the new wording: unanimous 7→13, MiMo agreement 83→96%, gpt-oss 50→67%.
  - run_code→read_file misses fell to 3 across 64 rows.
- **[2026-10-22] v3 wording for answer_directly / web_search / ask_user**, re-checked on all 64 candidates. Old sorts are in `tasks/tool_router/archive/` (v1, v2).
  - Checkers agree with each other more: 72% → 77%.
  - But the checkers agree with the generator less: unanimous 35 → 28, to-label 18 → 26.
  - Cause: the candidates' intended labels were written under the old wording.
  - Side effect: ask_user's "which file or item" phrase pulled 11 file/code requests into ask_user.
- **Next:** regenerate under the v3 wording; the generator reads `task.json`. Plan: Claude Haiku 5.5 for bulk and Sonnet 5.5 for edge cases, mixed with DeepSeek. The `ANTHROPIC_API_KEY` slot is in `.env`, and Justin pastes the key.
- **LLM backend config:** `.env` (gitignored) points at DeepInfra `deepseek-ai/DeepSeek-V4.1-Flash-1017`, and the key works.
  - First live run on the 22 seeds is in `runs/seeds-deepseek-v4.1-flash/`: DeepSeek 100% (p50 562 ms, p95 3.6 s) vs rules 63.6%.
  - The seeds are too easy to separate backends, so real or hard gold rows are needed.

## Done

- Research write-up in `docs/approach.md`: the zero-shot → distill → small-encoder ladder, what Jev and Simple Jev are, agent use and failure modes.
- `zsbench` harness (stdlib-only) with three baselines on one gold set:
  - `rules`
  - `jev` (Simple Jev `/v1/classifier`, with label-order rotation to cancel position bias)
  - `llm` (OpenAI-compatible JSON-schema output; optional Bearer key via `--llm-key-env`, e.g. DeepInfra)
- Report covers accuracy with a 95% bootstrap CI, macro-F1, invalid count, p50/p95 latency, ECE, order-flip rate and fixed-order accuracy, plus per-label F1 and confusion matrices.
- Gold-row spec (README, "What a gold row looks like"):
  - optional `context` turns
  - two-way `label` lists, where either label counts as correct
  - `dev`/`test` split
  - `note`
- `check` warns on more than 10% two-way rows and on near-duplicate gold rows. `check --against train.jsonl` finds gold leakage and exits 1 if there is any.
- Task `tasks/tool_router` with 22 synthetic seed rows: 1 with context, 1 two-way. The `label` CLI handles context and `2,3` two-way answers.
- Smoke run with no servers: rules scores 63.6% (CI 41–82) on the seeds, which is inflated since the rules were written against them. `jev` and `llm` skip cleanly.
- Prior-art survey in `docs/prior-art.md`:
  - who published training and data recipes (Kev, Laurer, OpenJev, Nimble, jevify, Laya, Clef, GLiClass)
  - license fit under the open-model rule
  - recipes to copy

## Not done / blockers

- No real model run yet. The Simple Jev and vLLM servers aren't installed (WSL, see README).
- Gold set is 22 synthetic rows; it needs 200 real agent inputs.
- The tool router is a placeholder decision; swap it for whichever decision the real agent needs.

## Decisions

- [2026-10-22] Claude API outputs are allowed for generation and teaching (Justin's call; this reverses the earlier no-Claude rule).
- The `llm` baseline uses the same Qwen as `jev`, so the comparison isolates scoring vs generating.
- Split about 50 dev / 150 test, since zero-shot baselines only tune descriptions and rules. Two-way rows are capped at 10%.
