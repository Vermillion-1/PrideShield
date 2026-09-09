# 05 — Future Work

*What this project would look like if rebuilt today, and what the current state of the art offers
that did not exist when it was built (July 2024 – May 2025).*

---

## Why this section exists

The pipeline in this repository — frozen CLIP embeddings, concatenation fusion, a small MLP head —
was a defensible design under 2024 tooling and a zero-dollar budget. It is **not** what anyone
should build now. The field moved, and the parts that moved are precisely the parts this project
struggled with: multimodal fusion, OCR on stylized text, and labeling throughput.

## Where the state of the art is now

### Vision-language models replaced the embed-then-classify pattern

The design here treats CLIP as a frozen feature extractor and learns a classifier on top. That was
standard practice in 2022–2024. It has largely been superseded by **instruction-tuned
vision-language models** that take an image and a text prompt and reason over both jointly:

| Model family | Relevance here |
|---|---|
| **Qwen2.5-VL / Qwen3-VL** | Strong open-weight VLMs with notably good OCR and text-in-image handling |
| **InternVL** | Competitive open-weight multimodal reasoning; strong on fine-grained visual detail |
| **LLaVA / LLaVA-NeXT** | The reference open architecture for visual instruction tuning |
| **Molmo, Pixtral, Gemma-3 vision** | Recent open-weight entrants with permissive licensing |
| **Frontier APIs** (Claude, GPT, Gemini multimodal) | Strongest zero-shot multimodal reasoning; no training required |

The relevant capability is not "better image classification" — it is that these models **read the
text inside the image and reason about its relationship to the imagery in one pass.** That is
precisely the operation concatenated CLIP embeddings cannot perform.

### Benchmarks now exist for this exact task

When this project started, hateful-meme detection had one main public benchmark
(**Meta's Hateful Memes Challenge**, 2020) plus scattered shared tasks. The space is better served
now — **HarMeme / Harm-P**, **MAMI** (misogynous memes), **HatReD** (hateful meme *reasoning* and
explanation), and multilingual efforts including **HOMO-MEX** for Spanish-language LGBTQ+ hate.

Practical implication: a restart should **evaluate on established public benchmarks first**, then
treat a custom corpus as a domain-specific supplement — the reverse of what we did.

### Parameter-efficient fine-tuning removed the compute wall

The single hardest constraint here was compute: a 307M-parameter encoder could not be fine-tuned on
free-tier Colab, which is why encoders stayed frozen. **LoRA / QLoRA** now make it feasible to
fine-tune multi-billion-parameter VLMs on a single consumer GPU by training small low-rank adapters
against a quantized base.

The "we can't afford to fine-tune" constraint that shaped this entire architecture is largely gone.

## The rebuild, in order

### Phase 1 — Establish the zero-shot baseline first

**Before training anything**, measure how a modern VLM performs zero-shot and few-shot on the
existing 1,320 examples. Report macro-averaged metrics and AUPRC.

This may end the project. If a zero-shot VLM matches or beats 93.94%, the trained pipeline has no
justification, and the honest conclusion is that the task no longer requires a custom model.

If it *doesn't*, this becomes the number every subsequent model must beat — which is a far more
meaningful bar than the 80.9% majority-class floor.

**Note this inverts the original workflow.** 2024: *collect data → train model → evaluate.*
2026: *evaluate what already exists → collect data only where it fails.*

### Phase 2 — Break the labeling bottleneck

Dataset size bounded nearly every result in this project, and labeling throughput bounded dataset
size.

Use a VLM to **pre-label at scale**, with humans adjudicating only low-confidence or disagreed
cases. The same human effort that produced 1,320 labels could plausibly cover an order of magnitude
more. Specific improvements to make while doing it:

- **Keep contested items** as soft labels or a dedicated hard-case evaluation set, rather than
  deleting them as we did.
- **Deliberately collect benign content** to approach realistic class balance instead of inheriting
  the 80.9/19.1 skew from a hate-focused search strategy.
- **Version the dataset** so results are reproducible against a specific snapshot.

### Phase 3 — Replace the fusion mechanism

If a trained model is still warranted after Phase 1:

- **Fine-tune a VLM directly** (LoRA/QLoRA) so the model attends jointly over image and caption
  natively. The fusion problem dissolves rather than being engineered around.
- If staying with frozen encoders for cost reasons, at minimum use **cross-attention fusion**
  instead of concatenation, so the two modalities can condition on each other.
- **Drop the OCR correction stack entirely** and let the VLM read the image text. Modern VLMs handle
  stylized, low-contrast, multi-panel meme text substantially better than 2024-era OCR, and can be
  prompted to preserve reading order. This removes one of the largest engineering components of the
  original pipeline.

### Phase 4 — Fix the evaluation protocol

The most important upgrades are methodological, not architectural:

- **A like-for-like text-only ablation** on the *same* corpus — the single fix that would make the
  central multimodal claim clean.
- **k-fold cross-validation with confidence intervals** — nearly free on cached embeddings, and the
  only way to establish whether the encoder gaps are real.
- **Macro-averaged metrics and AUPRC** as the headline figures, not positive-class metrics.
- **Multi-seed runs** with reported variance.
- **A test set at realistic class balance**, separate from the training distribution.
- **Confusion matrix and qualitative error analysis** — cheap, and the most informative analysis
  this project never did.

### Phase 5 — Evaluate what actually matters for deployment

None of this was measured, and all of it gates real use:

- **False-positive rate on reclaimed in-group language.** The dominant failure mode for this class
  of model, and the one where errors do the most harm — silencing the community the system exists to
  protect is worse than missing a slur. This deserves to be a primary metric, not an afterthought.
- **Adversarial robustness** — character substitution, leetspeak, text-in-image obfuscation, crops,
  and re-encoding. Hate speech adapts to filters by construction.
- **Subgroup breakdowns** across content types and target communities.
- **Calibration** — a moderation triage system needs trustworthy confidence scores to route the
  uncertain cases to humans.

### Phase 6 — Explanation, not just classification

A VLM can be prompted to **explain why** a meme is hateful, producing an auditable rationale rather
than a scalar. For moderation — where decisions are appealed and must be justified — this is far
more useful than a confidence value, and it directly addresses this project's interpretability gap.
The **HatReD** dataset targets exactly this reasoning task.

### Phase 7 — Infrastructure from day one

- **Experiment tracking** (W&B / MLflow) rather than metrics living in notebook cell outputs. Every
  number in [`04_evaluation.md`](04_evaluation.md) had to be reconstructed by cross-referencing the
  project report against surviving notebooks — that should never have been necessary.
- **Config-driven sweeps** instead of duplicated notebook cells.
- **Seeded, versioned runs** tied to dataset snapshots.

## What carries forward

Not everything here is superseded. These remain correct regardless of tooling:

- **Deduplication before splitting.** Near-duplicate leakage across train/test invalidates results
  no matter what model sits downstream. Perceptual hashing is still the right tool.
- **Benchmarking multiple encoders under identical downstream conditions.** The finding that
  `RN50x64` lost to a ViT with a quarter of its parameters only holds because the comparison was
  controlled.
- **Treating the loss function as a search parameter** rather than assuming Focal Loss suits
  imbalance. It was selected for one encoder and not another — an interaction that a fixed choice
  would have hidden.
- **Independent dual labeling** on a subjective task, and treating the disagreement rate as a
  measurement of task difficulty rather than an inconvenience.
- **The class-balance caveat.** Reporting accuracy against the 80.9% majority-class floor is a
  discipline that survives any architecture change.

---

**Previous:** [`04_evaluation.md`](04_evaluation.md) · **Back to:** [`../README.md`](../README.md)
