# Run report: seeds-deepseek-v4.1-flash

Task: `tool_router`, split: all, two-way rows (either label counts): 1 of 22

| backend | model | n | accuracy (95% CI) | macro-F1 | invalid | p50 ms | p95 ms | mean conf | ECE | order flips | acc, fixed order |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rules | - | 22 | 63.6% (41-82) | 0.641 | 0 | 0 | 0 | - | - | - | - |
| llm | deepseek-ai/DeepSeek-V4.1-Flash-1017 | 22 | 100.0% (100-100) | 1.000 | 0 | 562 | 3581 | - | - | - | - |

- **accuracy** uses the debiased prediction (Jev: averaged over label orders). The 95% CI is a bootstrap over rows; gaps between backends smaller than it are noise.
- **acc, fixed order** is Jev asked once with labels in task.json order, i.e. a naive single call.
- **order flips** is the share of inputs where Jev's choice changed with label order (position bias).
- **ECE** compares stated confidence with real accuracy; lower is better. Simple Jev's probabilities are not calibrated.

## F1 per label

A rough guide when support is under about 30: one or two mistakes move these a lot.

| label | support | rules | llm |
|---|---|---|---|
| answer_directly | 4 | 0.500 | 1.000 |
| web_search | 4 | 0.667 | 1.000 |
| read_file | 5 | 0.889 | 1.000 |
| run_code | 5 | 0.750 | 1.000 |
| ask_user | 4 | 0.400 | 1.000 |

## Confusion: rules (rows = gold, cols = predicted)

| gold \ pred | answer_directly | web_search | read_file | run_code | ask_user |
|---|---|---|---|---|---|
| answer_directly | 4 | 0 | 0 | 0 | 0 |
| web_search | 2 | 2 | 0 | 0 | 0 |
| read_file | 1 | 0 | 4 | 0 | 0 |
| run_code | 2 | 0 | 0 | 3 | 0 |
| ask_user | 3 | 0 | 0 | 0 | 1 |

## Confusion: llm (rows = gold, cols = predicted)

| gold \ pred | answer_directly | web_search | read_file | run_code | ask_user |
|---|---|---|---|---|---|
| answer_directly | 4 | 0 | 0 | 0 | 0 |
| web_search | 0 | 4 | 0 | 0 | 0 |
| read_file | 0 | 0 | 5 | 0 | 0 |
| run_code | 0 | 0 | 0 | 5 | 0 |
| ask_user | 0 | 0 | 0 | 0 | 4 |
