# Zero-shot / few-shot classifier: approach

[2026-10-22] Research notes. Jev and Simple Jev shipped after Claude's training cutoff, so everything about them below comes from Oct 2026 web sources. Treat their benchmark numbers as unverified until you rerun them on your own data.

## The answer

Don't train a foundation model. Build a thin, three-stage ladder, then move each task down it as you get data:

1. **Zero-shot (day 0, no labels):** score each candidate label with an LLM's logits. Don't have the LLM generate JSON.
2. **Distill (once you have traffic):** label a few hundred to a few thousand inputs with teacher models. Hand-check about 200 of them as a gold eval set.
   - [2026-10-22] Teachers and generators are DeepInfra open models plus the Claude API (Haiku 5.5 bulk, Sonnet 5.5 edge cases), mixed across generators.
   - Justin reversed the earlier no-Claude rule.
3. **Few-shot / production (fixed task):** train a small encoder (SetFit, or ModernBERT/MiniLM + logistic regression) on that data. It runs in about 8 ms on CPU and usually beats every zero-shot option.

Keep stage 1 running for new or changing labels. Fixed labels move to stage 3.

## What Jev actually is

- **Jev (TypeSafe AI, closed API, ~Sep 2026):** a "System One" decision model. You send a state (text or JSON) and typed questions:
  - `choice`: pick from up to 255 options
  - `score`: rate against a rubric
  - `noul`: probability that a statement is true

  It returns probabilities and confidence, never prose. TypeSafe hasn't published the architecture, size or weights. Outside analysis says it reads the score of each allowed answer, the same multiple-choice trick used in LLM evals and reward models.
- **Simple Jev (Featherless, Apache 2.0):** an open clone of the same contract. It wraps Hugging Face models (Qwen, Gemma) and scores labels from next-token logits, so outputs never fail to parse. You can self-host it.
- **Other open options mentioned:** Laya (small encoders with a choice head, about 322M–421M params) and Clef (frozen Qwen + LoRA + a scoring head, Apache 2.0). Neither has been verified here.

## How this improves smaller models

The trick isn't the model, it's the output head. A small open LLM asked to *generate* a label is slow, can't parse its own output reliably, and is badly calibrated. The same model *scoring* a fixed label set gives you:

- one forward pass, no decoding
- no parse failures
- a real probability distribution you can threshold

That's why Simple Jev on Qwen ~27B lands close to closed Jev on one leaderboard (55.7 vs 57.9 on the "Decision Index", 373 ms vs 524 ms).

The second lever is **decomposition**. Split a hard label into atomic yes/no questions and feed those scores to a tiny classifier. On phishing detection, asking Jev directly got about 60%; 8 indicator questions plus a lightweight classifier got 94%.

## Reported numbers (from single sources, unverified)

| Task | Zero-shot Jev | Old NLI zero-shot | Trained small model |
|---|---|---|---|
| Banking77, 77 intents | 80.1% | 48.8% (66.7% newer NLI) | 93.2% (22M encoder + LR, 8 ms CPU) |
| Yelp | 67.2% | n/a | 51.9% |
| Glean query classification (agreement with an LLM) | 66.8% | n/a | 74.5% (fine-tuned Laya) |
| Glean routing, 3 experts | 97.6% P / 98.7% R | n/a | LLM router: 95.6% / 93.4%, 8x slower |

The pattern: zero-shot wins when you have no labels or the labels keep changing. A trained small model wins on a fixed task.

## Using it in an agent

Good fits:

- Routing: which tool, which model, which sub-agent
- Gates: "is this task done?", "is this on-topic?", injection and content filters
- Reranking candidates
- Judging: "does this passage support the claim?"

Bad fits:

- Generating tool arguments. A generative model still has to write the query or code.
- Math, dates, business rules. Keep those in code.
- The sole gate on destructive, financial or personal-data actions. One appended unverified claim flipped 12% of decisions in one test.

Known failure modes to test against:

- **Position bias:** it picked the first option in 3,000/3,000 random draws.
- **Label wording:** switching "0/1" to "no/yes" flipped 70 of 100 answers.
- **Overconfidence:** about 88% stated confidence vs about 80% accuracy on Banking77. Recalibrate on your own data before you threshold on it.
- **Large label sets and out-of-distribution tasks:** it loses to specialized classifiers there.

Mitigations:

- Shuffle option order and average the results.
- Mark tool output, email and web pages as untrusted.
- Pin the model version.

## Build vs buy

| Option | Use when |
|---|---|
| Jev API | You want zero hosting, data can leave the box, and you're prototyping. Never use its outputs as training labels. |
| Simple Jev + local Qwen/Gemma | You want privacy, offline use and fixed latency, or to run it inside a local agent. **Default pick for us.** |
| Roll your own scorer | About 100 lines: feed the prompt and each label, read the log-prob, softmax over the labels. Do this if Simple Jev's model support doesn't fit. |
| Trained encoder (SetFit / ModernBERT) | The task is fixed and you have 50+ labeled examples per class (including teacher-labeled). Cheapest and most accurate. |

## Suggested first build

1. Pick one real agent decision (for example, the tool router) and write 200 gold examples.
2. Run three baselines on those 200. This is built in `zsbench/` (see the README).
   - Simple Jev on a local Qwen
   - the same Qwen generating JSON
   - a keyword/rule baseline
3. Measure accuracy, p50 latency and calibration. Check shuffled vs fixed option order.
4. Take the best open-weight backend as the teacher. Label about 2k inputs and train a SetFit/ModernBERT student. Compare it on the same 200.

## Sources

- [Parallel: Testing Jev](https://parallel.ai/blog/testing-jev)
- [TypeSafe docs](https://docs.typesafe.ai)
- [Glean: Jev and the return of the zero-shot classifier](https://www.glean.com/blog/jev-zero-shot-classifier)
- [MindStudio: Jev vs classic classifiers](https://www.mindstudio.ai/blog/jev-vs-classic-classifiers-benchmark/). Note that MindStudio sells a competing product.
- [Akita: Understanding the Jev hype](https://akitaonrails.com/en/2026/10/03/understanding-the-jev-hype-what-is-it-for/)
- [AlphaSignal: Featherless Simple Jev](https://alphasignal.ai/news/featherless-ai-s-simple-jev-turns-open-hugging-face-models-into-classifier-apis)
- [Jev in the Wild (arXiv 2609.30216)](https://arxiv.org/abs/2609.30216): 2,170 GitHub projects; routing and agent use cases get the most attention.
- [Kunal Ganglani: Jev vs Laya](https://www.kunalganglani.com/blog/jev-vs-laya-head)
- [Zeli / Raschka thread](https://zeli.app/story/49891203)
