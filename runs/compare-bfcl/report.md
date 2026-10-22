# Run report: compare-bfcl

Task: `tool_router_bfcl`, split: all, two-way rows (either label counts): 0 of 224

Scope: test split

| backend | model | n | accuracy (95% CI) | macro-F1 | invalid | p50 ms | p95 ms | mean conf | ECE | order flips | acc, fixed order |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rules | - | 224 | 46.0% (40-52) | 0.268 | 0 | 0 | 0 | - | - | - | - |
| llm | deepseek-ai/DeepSeek-V4.1-Flash-1017 | 224 | 78.6% (73-84) | 0.464 | 0 | 887 | 5319 | - | - | - | - |
| intern | intern | 224 | 75.4% (70-81) | 0.553 | 0 | 416 | 468 | 62.5% | 0.135 | 34.8% | 68.8% |
| kev | kev-latest | 224 | 70.5% (65-76) | 0.520 | 0 | 283 | 305 | 56.0% | 0.149 | 17.9% | 68.3% |

- **accuracy** uses the debiased prediction (Jev: averaged over label orders). The 95% CI is a bootstrap over rows; gaps between backends smaller than it are noise.
- **acc, fixed order** is Jev asked once with labels in task.json order, i.e. a naive single call.
- **order flips** is the share of inputs where Jev's choice changed with label order (position bias).
- **ECE** compares stated confidence with real accuracy; lower is better. Simple Jev's probabilities are not calibrated.

## F1 per label (test split)

A rough guide when support is under about 30: one or two mistakes move these a lot.

| label | support | rules | llm | intern | kev |
|---|---|---|---|---|---|
| answer_directly | 32 | 0.339 | 0.674 | 0.610 | 0.564 |
| web_search | 96 | 0.274 | 0.798 | 0.766 | 0.779 |
| read_file | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| run_code | 96 | 0.728 | 0.847 | 0.834 | 0.736 |
| ask_user | 0 | 0.000 | 0.000 | 0.000 | 0.000 |

Confusion matrices and misses are hidden: this run has only test rows. Run with the dev split to see errors you can tune on.
