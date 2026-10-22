# Project: ZS-FewShot

Zero-shot and few-shot classifiers for agent decisions (routing, gates, judging). The plan:

1. Score labels with a local open model (Simple Jev).
2. Distill into a small encoder once a task is fixed.

## Features

- [2026-10-22] `zsbench`: baseline harness comparing rules, Simple Jev and LLM JSON generation on a gold set, with a markdown report and a hand-labeling CLI.
- [2026-10-22] `generate` / `blindcheck`:
  - DeepSeek generates candidates.
  - gpt-oss-120b and MiMo check them blind; Llama-4-Scout breaks ties.
  - Output goes to `teacher.jsonl` (training data) or `to_label.jsonl` (hand-label into gold).

## Decisions

- [2026-10-22] **No local model servers on this machine's GPU** (shared with PROD AURA). `gpu_guard.py launch` is locked out (exit 4), and the README launch commands are removed. Lifting it is a code edit on a GPU without PROD.
- [2026-10-22] **The router is DeepSeek V4.1 Flash on DeepInfra** (`zsbench/router.py`, `python -m zsbench route`). Justin's pick: best on BFCL (78.6% vs Intern 75.4% and Kev 70.5%), tied on our test split, and no load on the PROD GPU. Fallback is `answer_directly` after one retry.

- [2026-10-22] Generators and teachers: DeepInfra open models plus the Claude API (Haiku 5.5 bulk, Sonnet 5.5 edge cases), mixed across generators. This supersedes the earlier "open-weight only, no Claude" rule; Justin reversed it.
- [2026-10-22] "Read a file then compute" = `run_code`. `read_file` = reading is the answer.
- [2026-10-22] Gold is split dev/test (`zsbench split`). Tune on dev only and report test; the report never lists individual test errors.
- [2026-10-22] tool_router wording is frozen at v1 (fingerprint `770dd97d3a97806e`, enforced by a test). At about 27 test rows, wording changes can't be measured, so the next lever is more real gold rows.
- [2026-10-22] The harness has zero dependencies (stdlib HTTP), and the model servers run separately in WSL. This keeps the harness portable and the comparison server-agnostic.
- [2026-10-22] Jev debiasing: send one question per cyclic rotation of the label list in a single request, and average the probabilities. Simple Jev assigns A/B/C letters in criteria order, so position bias is real; the shared prefill makes the extra questions cheap.

## Tech debt

- Serving commands in the README come from Simple Jev's README and are not yet run on this machine.
- `--llm-extra` defaults to vLLM's `chat_template_kwargs`; other servers may reject it.
