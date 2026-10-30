# Run report: snapchat-test

Task: `tool_router`, split: test, two-way rows (either label counts): 3 of 27

Scope: test split

| backend | model | n | accuracy (95% CI) | macro-F1 | invalid | p50 ms | p95 ms | mean conf | ECE | order flips | acc, fixed order |
|---|---|---|---|---|---|---|---|---|---|---|---|
| snapchat | snapchat-my-ai | 27 | 92.6% (81-100) | 0.918 | 0 | 1518 | 1788 | - | - | - | - |

- **accuracy** uses the debiased prediction (Jev: averaged over label orders). The 95% CI is a bootstrap over rows; gaps between backends smaller than it are noise.
- **acc, fixed order** is Jev asked once with labels in task.json order, i.e. a naive single call.
- **order flips** is the share of inputs where Jev's choice changed with label order (position bias).
- **ECE** compares stated confidence with real accuracy; lower is better. Simple Jev's probabilities are not calibrated.

## F1 per label (test split)

A rough guide when support is under about 30: one or two mistakes move these a lot.

| label | support | snapchat |
|---|---|---|
| answer_directly | 6 | 1.000 |
| web_search | 5 | 1.000 |
| read_file | 7 | 0.933 |
| run_code | 6 | 0.800 |
| ask_user | 3 | 0.857 |

Confusion matrices and misses are hidden: this run has only test rows. Run with the dev split to see errors you can tune on.
