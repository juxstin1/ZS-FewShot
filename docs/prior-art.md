# Prior art: who published their training setup and data pipeline

[2026-10-22] Web research. Most of these projects are from Sep–Oct 2026, after Claude's training cutoff. Each finding comes from the project's own README or model card; nothing here has been reproduced. The license column is checked against our rule: no outputs from Claude, GPT, closed Jev, or other closed APIs as training data.

## Who exposed the most

| project | base / size | what's public | data generation | license fit for us |
|---|---|---|---|---|
| **Kev** (Jared Palmer) | Qwen3.5 0.8B/4B/9B (LoRA r16 + pointer head); 27B full fine-tune | Everything: code, recipe, ~29 logged experiment rounds in `PLAN.md`, frozen eval suites with checksums | 10 public datasets + generated policy/rule examples. **Labels only from open-weight teachers (GLM-5.3, DeepSeek-V4-Pro, Inkling) or code solvers.** Kept only when 2 blind labelers agreed with the intended answer. | Best fit. Apache code and weights; datasets keep their own licenses. "No Jev outputs were used." |
| **Laurer zeroshot v1/v2** (DeBERTa/ModernBERT NLI) | DeBERTa-v3, 184M–435M | Paper + 6 notebooks: harmonize → clean → format → train → eval | v1: 33 human-labeled datasets. v2 "-c": synthetic texts from **Mixtral (Apache)** seeded by 500+ curated task ideas | Good. The "-c" data is commercially clean. |
| **Kev-0.5B prototype** | Qwen2.5-0.5B frozen, LoRA r16 | Full card: data mix, augmentation, hyperparameters, results | 6 public datasets × 1,500 records converted to choice/score/noul. **No LLM-generated data.** | Good |
| **OpenJev-8B** (UESTC) | Qwen3-8B-Base, LoRA r16 + pointer | Code, data builders, mixing ratios, paper | Mind2Web + live-web steps, teacher episodes from **qwen3-max** (closed API), LLM-judge relabeling | Mixed. qwen3-max outputs, research-only tasksource tasks, and a "jev-distill" corpus. Don't train on its data. |
| **Bespoke Nimble** | Qwen3.5-9B, LoRA r16 | Detailed curation guide | Contrastive pairs: edit ≤8 words to flip the answer, then a deletion test proves the evidence matters. **Generated with GPT-5.6 and Claude.** | Copy the method, not the data. |
| **jevify** | Gemma-4-26B-A4B, LoRA r64 | Card: ~47k items, KL loss on human label distributions | Hard-label sets + multi-annotator sets with human label distributions | Gemma license, not Apache |
| **Laya** (ConvAI) | ModernBERT-large, 421M | Architecture, RL loss, a Kaggle fine-tune notebook | Not described on the card (the notebook builds a dataset) | Apache, but data unclear |
| **Clef** (Cloudflare) | Qwen3.8-27B / 9B + scoring head | Architecture and losses only | **Internal synthetic data, not public** | Weights only |
| **GLiClass** (Knowledgator) | ModernBERT / DeBERTa | Paper; cards say "synthetic and licensed data that allow commercial use" | Not detailed | Unclear |

## Recipes worth stealing

### Data generation (under our open-model rule)

- **Kev's double-blind filter.** An open model generates an example with an intended answer. Two separate open-model labelers answer it blind, and the example is kept only if both match. This is the cleanest synthetic-label gate published.
- **Nimble's contrastive pairs.** Make two near-identical examples where a single fact flips the label (edit at most 8 words). Then delete each evidence sentence in turn: the answer must become unknowable. This teaches the model *which* evidence matters. Use open models for it, not GPT or Claude.
- **Laurer's seeded diversity.** Brainstorm 500+ task ideas first, curate the list by hand, then generate texts per task. Diversity comes from the seed list, not from sampling temperature.
- **Programmatic labels wherever possible.** Kev uses code solvers for dates, counts and rules, which matches the Jev docs' warning that it "is not a calculator."

### Cleaning and decontamination

- **Laurer: CleanLab on SentenceBERT embeddings** dropped about 17% of texts as likely mislabeled. Cap at 500 per class and 5,000 per dataset: "data diversity and quality is more important than quantity."
- **Kev's screen** dropped 6,581 records, including 1,956 that matched on question wording alone, which a template filter missed. Also run exact and near-dup checks against your own dev/test sets (`zsbench check --against` does this).
- **Audit the eval labels too.** Kev threw out benchmark sets whose gold was the problem:
  - WANLI: annotators disagreed on 27% of pairs.
  - Several tasksource families: 21–55% "all-wrong".
  - Keyword-labeled emotion data.
  - TypeSafe's own evals, whose gold is just two frontier models' averaged answer.

  This is the reason for our two-way labels and the "don't drop hard rows" rule.
- **Template overlap inflates scores.** Kev's hard-v1 eval shared training templates, and it carried most of one release's net gain. Group splits by source/template family, not by row.

### Augmentation (for when we train)

From Kev-0.5B, all applied once per record:

- Shuffle option order.
- About 30% of option descriptions set to null.
- About 32% of states wrapped as JSON.
- 10% of the time, replace the true option with "none of the above".
- 15% of the time, add a distractor option.

Kev keeps a 25% none-of-the-above pair rate because the model needs to learn "unknowable".

### Training

- **LoRA r16 on a frozen base plus a small scoring head** is the common recipe: Kev, OpenJev and Nimble all use it, with cross-entropy over the allowed options.
- **Learning rate:** 2e-4 "erased much pre-trained knowledge" while 5e-5 preserved it (Kev 4B/9B). Run 1–2 epochs.
- **Soft labels beat hard labels on ambiguous items** (targets at p ≥ 0.6). jevify trains with KL against human label distributions.
- **Diminishing returns from one generator:** the first 6k records gave +26 points, the next 12k gave +5. Mix sources and teachers.
- **Replay public data** to avoid forgetting at small sizes. At 9B and up it stopped helping.
- **Didn't help (Kev):**
  - calibration losses (label smoothing, Brier, focal)
  - permutation-consistency KL
  - weight interpolation with the base

### Calibration

- **Fit one global temperature on held-out *datasets*, not held-out rows from training data.** In-distribution temperatures failed to transfer: ECE 0.059 vs 0.0085.
- Per-(question type, option count) temperatures were worse for Kev. Laya uses them anyway: ECE 0.466 → 0.081.

### Evaluation design

- **Laurer: leave-one-dataset-out.** Train 28 models, each missing one dataset, and score on the missing one. This is the honest way to claim "zero-shot".
- **Kev:**
  - frozen eval suites with checksummed files
  - checkpoints picked on dev
  - each test set read once per released model
  - a separate "transfer" suite of sources never seen in training

## What this means for our next steps

1. **Teacher for stage 2:** follow Kev. Use 2+ open-weight models (Qwen, DeepSeek, GLM), keep only rows where blind labelers agree, and use code solvers for anything computable.
2. **Hard cases for the gold set:** use Nimble-style contrastive pairs generated with open models. They're also a cheap way to write "unknowable" rows for `ask_user`.
3. **Before training anything,** read Kev's `PLAN.md` negative results. It's ~29 rounds of what didn't work, and it'll save weeks.

## Sources

- [Kev repo + PLAN.md](https://github.com/jaredpalmer/kev) · [Kev-0.5B card](https://huggingface.co/Terom/kev-0.5b)
- [Laurer: Building Efficient Universal Classifiers with NLI (arXiv 2312.17543)](https://arxiv.org/html/2312.17543v2) · [notebooks](https://github.com/MoritzLaurer/zeroshot-classifier) · [Mixtral synthetic dataset](https://huggingface.co/datasets/MoritzLaurer/synthetic_zeroshot_mixtral_v0.1)
- [OpenJev-8B card](https://huggingface.co/alanhuangya/OpenJev-8B) · [OpenJev repo](https://github.com/alanhuangyoo/OpenJev)
- [Bespoke Nimble repo](https://github.com/bespokelabsai/nimble) · [curation guide](https://github.com/bespokelabsai/nimble/blob/main/docs/TRAINING_EVAL_CURATION.md)
- [jevify-gemma4-26b-a4b](https://featherless.ai/models/kushalpatil/jevify-gemma4-26b-a4b)
- [Laya card](https://huggingface.co/convaiinnovations/laya) · [Laya repo](https://github.com/NandhaKishorM/laya)
- [Cloudflare Clef (The Register: training data not public)](https://www.theregister.com/a/5300649) · [Clef summary](https://letsdatascience.com/news/cloudflare-releases-open-weight-clef-decision-models-c52886d6)
- [GLiClass paper (arXiv 2508.07662)](https://arxiv.org/abs/2508.07662v1) · [gliclass-modern-base-v2.0 card](https://huggingface.co/knowledgator/gliclass-modern-base-v2.0)
- [Jev's open-source clones roundup](https://www.besthub.dev/articles/jev-s-open-source-clones-arrive-in-48-hours-9b-nimble-0-5b-kev-711c194243e9)
