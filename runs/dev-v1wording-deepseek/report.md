# Run report: dev-v1wording-deepseek

Task: `tool_router`, split: dev, two-way rows (either label counts): 0 of 15

Scope: dev split

| backend | model | n | accuracy (95% CI) | macro-F1 | invalid | p50 ms | p95 ms | mean conf | ECE | order flips | acc, fixed order |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llm | deepseek-ai/DeepSeek-V4.1-Flash-1017 | 15 | 100.0% (100-100) | 1.000 | 0 | 606 | 1535 | - | - | - | - |

- **accuracy** uses the debiased prediction (Jev: averaged over label orders). The 95% CI is a bootstrap over rows; gaps between backends smaller than it are noise.
- **acc, fixed order** is Jev asked once with labels in task.json order, i.e. a naive single call.
- **order flips** is the share of inputs where Jev's choice changed with label order (position bias).
- **ECE** compares stated confidence with real accuracy; lower is better. Simple Jev's probabilities are not calibrated.

## F1 per label (dev split)

A rough guide when support is under about 30: one or two mistakes move these a lot.

| label | support | llm |
|---|---|---|
| answer_directly | 4 | 1.000 |
| web_search | 3 | 1.000 |
| read_file | 3 | 1.000 |
| run_code | 3 | 1.000 |
| ask_user | 2 | 1.000 |

## Confusion: llm (dev only; rows = gold, cols = predicted)

| gold \ pred | answer_directly | web_search | read_file | run_code | ask_user |
|---|---|---|---|---|---|
| answer_directly | 4 | 0 | 0 | 0 | 0 |
| web_search | 0 | 3 | 0 | 0 | 0 |
| read_file | 0 | 0 | 3 | 0 | 0 |
| run_code | 0 | 0 | 0 | 3 | 0 |
| ask_user | 0 | 0 | 0 | 0 | 2 |

## Dev misses (tune wording on these)

Test misses are left out on purpose: looking at them turns test into another dev set.

### llm: 0 misses

