# Run report: compare-test

Task: `tool_router`, split: test, two-way rows (either label counts): 3 of 27

Scope: test split

| backend | model | n | accuracy (95% CI) | macro-F1 | invalid | p50 ms | p95 ms | mean conf | ECE | order flips | acc, fixed order |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rules | - | 27 | 55.6% (37-74) | 0.571 | 0 | 0 | 0 | - | - | - | - |
| llm | deepseek-ai/DeepSeek-V4.1-Flash-1017 | 27 | 88.9% (74-100) | 0.885 | 0 | 1380 | 5733 | - | - | - | - |
| intern | intern | 27 | 88.9% (74-100) | 0.876 | 0 | 423 | 471 | 60.8% | 0.281 | 33.3% | 66.7% |
| kev | kev-latest | 27 | 88.9% (74-100) | 0.876 | 0 | 260 | 280 | 52.4% | 0.365 | 18.5% | 85.2% |
| snapchat | snapchat-my-ai | 27 | 92.6% (81-100) | 0.918 | 0 | 1518 | 1788 | - | - | - | - |

- **accuracy** uses the debiased prediction (Jev: averaged over label orders). The 95% CI is a bootstrap over rows; gaps between backends smaller than it are noise.
- **acc, fixed order** is Jev asked once with labels in task.json order, i.e. a naive single call.
- **order flips** is the share of inputs where Jev's choice changed with label order (position bias).
- **ECE** compares stated confidence with real accuracy; lower is better. Simple Jev's probabilities are not calibrated.

## F1 per label (test split)

A rough guide when support is under about 30: one or two mistakes move these a lot.

| label | support | rules | llm | intern | kev | snapchat |
|---|---|---|---|---|---|---|
| answer_directly | 6 | 0.455 | 1.000 | 0.923 | 0.857 | 1.000 |
| web_search | 5 | 0.571 | 1.000 | 0.889 | 0.800 | 1.000 |
| read_file | 7 | 0.727 | 0.857 | 0.933 | 1.000 | 0.933 |
| run_code | 6 | 0.600 | 0.769 | 0.833 | 0.923 | 0.800 |
| ask_user | 3 | 0.500 | 0.800 | 0.800 | 0.800 | 0.857 |

Confusion matrices and misses are hidden: this run has only test rows. Run with the dev split to see errors you can tune on.
