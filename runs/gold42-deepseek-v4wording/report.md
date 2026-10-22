# Run report: gold42-deepseek-v4wording

Task: `tool_router`, split: all, two-way rows (either label counts): 3 of 42

| backend | model | n | accuracy (95% CI) | macro-F1 | invalid | p50 ms | p95 ms | mean conf | ECE | order flips | acc, fixed order |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rules | - | 42 | 57.1% (43-71) | 0.542 | 0 | 0 | 0 | - | - | - | - |
| llm | deepseek-ai/DeepSeek-V4.1-Flash-1017 | 42 | 92.9% (83-100) | 0.928 | 0 | 631 | 1011 | - | - | - | - |

- **accuracy** uses the debiased prediction (Jev: averaged over label orders). The 95% CI is a bootstrap over rows; gaps between backends smaller than it are noise.
- **acc, fixed order** is Jev asked once with labels in task.json order, i.e. a naive single call.
- **order flips** is the share of inputs where Jev's choice changed with label order (position bias).
- **ECE** compares stated confidence with real accuracy; lower is better. Simple Jev's probabilities are not calibrated.

## F1 per label

A rough guide when support is under about 30: one or two mistakes move these a lot.

| label | support | rules | llm |
|---|---|---|---|
| answer_directly | 10 | 0.529 | 0.909 |
| web_search | 8 | 0.545 | 1.000 |
| read_file | 10 | 0.778 | 1.000 |
| run_code | 9 | 0.571 | 0.842 |
| ask_user | 5 | 0.286 | 0.889 |

## Confusion: rules (rows = gold, cols = predicted)

| gold \ pred | answer_directly | web_search | read_file | run_code | ask_user |
|---|---|---|---|---|---|
| answer_directly | 9 | 0 | 0 | 1 | 0 |
| web_search | 5 | 3 | 0 | 0 | 0 |
| read_file | 3 | 0 | 7 | 0 | 0 |
| run_code | 3 | 0 | 1 | 4 | 1 |
| ask_user | 4 | 0 | 0 | 0 | 1 |

## Confusion: llm (rows = gold, cols = predicted)

| gold \ pred | answer_directly | web_search | read_file | run_code | ask_user |
|---|---|---|---|---|---|
| answer_directly | 10 | 0 | 0 | 0 | 0 |
| web_search | 0 | 8 | 0 | 0 | 0 |
| read_file | 0 | 0 | 9 | 0 | 0 |
| run_code | 2 | 0 | 0 | 8 | 0 |
| ask_user | 0 | 0 | 0 | 1 | 4 |
