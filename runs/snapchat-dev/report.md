# Run report: snapchat-dev

Task: `tool_router`, split: dev, two-way rows (either label counts): 0 of 15

Scope: dev split

| backend | model | n | accuracy (95% CI) | macro-F1 | invalid | p50 ms | p95 ms | mean conf | ECE | order flips | acc, fixed order |
|---|---|---|---|---|---|---|---|---|---|---|---|
| snapchat | snapchat-my-ai | 15 | 80.0% (60-100) | 0.776 | 0 | 1532 | 3535 | - | - | - | - |

- **accuracy** uses the debiased prediction (Jev: averaged over label orders). The 95% CI is a bootstrap over rows; gaps between backends smaller than it are noise.
- **acc, fixed order** is Jev asked once with labels in task.json order, i.e. a naive single call.
- **order flips** is the share of inputs where Jev's choice changed with label order (position bias).
- **ECE** compares stated confidence with real accuracy; lower is better. Simple Jev's probabilities are not calibrated.

## F1 per label (dev split)

A rough guide when support is under about 30: one or two mistakes move these a lot.

| label | support | snapchat |
|---|---|---|
| answer_directly | 4 | 0.857 |
| web_search | 3 | 1.000 |
| read_file | 3 | 0.857 |
| run_code | 3 | 0.500 |
| ask_user | 2 | 0.667 |

## Confusion: snapchat (dev only; rows = gold, cols = predicted)

| gold \ pred | answer_directly | web_search | read_file | run_code | ask_user |
|---|---|---|---|---|---|
| answer_directly | 3 | 0 | 0 | 0 | 1 |
| web_search | 0 | 3 | 0 | 0 | 0 |
| read_file | 0 | 0 | 3 | 0 | 0 |
| run_code | 0 | 0 | 1 | 1 | 1 |
| ask_user | 0 | 0 | 0 | 0 | 2 |

## Dev misses (tune wording on these)

Test misses are left out on purpose: looking at them turns test into another dev set.

### snapchat: 3 misses

| id | gold | predicted | text |
|---|---|---|---|
| 9979e8ef4c0c | answer_directly | ask_user | here's the job posting i'm applying to, pasted below. how many years of experience do they actually  |
| a48b0535ff4e | run_code | read_file | Can you check whether my thermostat schedule script in /home/me/schedules.py actually produces valid |
| 09d3a9d022c6 | run_code | ask_user | rebalance it |

