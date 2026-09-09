# 06 — Retrospective: This Project in 2024 vs. Now

*Written after the fact. An honest assessment of which decisions were right for their moment, which
were wrong at the time, and how this would be built today.*

---

## What this project was, in context

Built July 2024 – May 2025 as an undergraduate major project, with **no budget** for APIs,
annotation, or compute. The design space was genuinely constrained, and most architectural decisions
follow from that rather than from a free choice among alternatives.

The core hypothesis — *multimodal signal beats text-only for meme hate speech* — was sound, and the
results support it directionally.

## Decisions that were right for 2024

**Frozen CLIP embeddings + small trainable head.**
Forced by free-tier compute, but genuinely the correct choice under the constraint. It made a
30-trial hyperparameter search and a three-encoder comparison affordable, and it makes the work
reproducible in minutes.

**Benchmarking three encoders rather than picking one.**
Easy to skip; we didn't. It produced the project's most interesting finding — the CNN losing to a
much smaller ViT.

**Making the loss function a search parameter.**
Rather than assuming Focal Loss was right for imbalance, we let Optuna choose between it and
class-weighted CE. It selected differently per encoder, which we would not otherwise have learned.

**Implementing Focal Loss from scratch.**
Not strictly necessary, but it meant understanding the mechanism rather than importing it.

**Building the OCR correction stack.**
T5 + Levenshtein repair addressed the actual quality bottleneck. High-leverage, if unmeasured.

## Decisions that were wrong at the time

These aren't hindsight-with-better-tools — they were mistakes by 2024's own standards:

**1. Dropping contested labels instead of keeping them.**
Deleted exactly the hard cases, inflating every metric. Soft labels or a separate hard-case
evaluation set were standard practice and available.

**2. Training the text baseline on a different corpus.**
FRENK instead of meme captions makes the central comparison confounded. The fix was cheap and
obvious.

**3. No k-fold cross-validation.**
On ~1,320 examples with cached embeddings, k-fold costs minutes. A single fixed split with a
198-example test set was the wrong protocol.

**4. Reporting positive-class rather than macro metrics.**
On an 80.9/19.1 split, positive-class precision/recall flatter the result. Macro should have led.

**5. No error analysis.**
The cheapest, most informative analysis available — *which memes does it get wrong?* — was never
done. A confusion matrix and fifty misclassified examples would have taught more than another
architecture.

**6. No experiment tracking.**
Results lived in notebook outputs and report tables, which is why reconstructing them later required
cross-referencing multiple sources. W&B existed and was free.

## What 2026 changes

The tooling landscape shifted underneath this project between when it was built and now.

### Vision-language models replace the entire architecture

In 2024, CLIP embeddings + a classifier head was a reasonable multimodal design. It no longer is.
Instruction-tuned VLMs — LLaVA, Qwen-VL, InternVL, or a frontier multimodal API — can be evaluated
**zero-shot on this task with no training at all**.

**Any restart should begin by measuring a zero-shot VLM baseline**, because:
- It may beat the trained CLIP+MLP pipeline outright, which would make the trained model
  unjustifiable.
- If it doesn't, it establishes what fine-tuning must beat to be worth doing.
- It requires no labeled data, sidestepping the project's binding constraint.

This inverts the original workflow: 2024 said *collect data → train model*; 2026 says *measure
zero-shot → collect data only where the model fails*.

### Native fusion replaces concatenation

Concatenating frozen embeddings cannot represent the *interaction* between image and caption — which
is where meme hostility lives. A VLM attends jointly over image and text natively, so the fusion
problem dissolves rather than being engineered around.

### VLMs read meme text better than 2024 OCR

Modern VLMs handle stylized, low-contrast, multi-panel meme text substantially better than EasyOCR,
and can be prompted to preserve reading order. **This would likely eliminate the entire OCR
correction stack** — arguably the largest engineering component of the original project — as a
side effect.

### Explainability becomes possible

A VLM can be prompted to *explain why* a meme is hateful, producing an auditable rationale rather
than a scalar score. For content moderation — where decisions get appealed and must be justified —
this is far more useful than a confidence value, and it directly addresses the interpretability gap
in the original system.

### The annotation bottleneck breaks

The binding constraint was human labeling capacity. VLM-assisted pre-labeling with human
adjudication of low-confidence cases would plausibly have yielded **an order of magnitude more
data** for the same human effort. Since nearly every failed experiment here traced back to dataset
size, this is the change with the largest expected effect.

## How I'd rebuild it today

**Phase 1 — Establish the real baseline.**
Zero-shot and few-shot VLM evaluation on the existing 1,320 examples. Report macro metrics and
AUPRC. This may conclude the project: if zero-shot is sufficient, there's nothing to train.

**Phase 2 — Scale the data.**
VLM pre-labeling across a much larger scrape; humans adjudicate only contested items. Retain
disagreements as soft labels or a hard-case benchmark. Compute κ. Deliberately collect benign
LGBTQ+ content to approach realistic class balance.

**Phase 3 — Train only if justified.**
If zero-shot underperforms, fine-tune a VLM (LoRA/QLoRA on consumer hardware) against it. Every
trained model must beat the zero-shot number to earn its place.

**Phase 4 — Evaluate properly.**
k-fold CV with confidence intervals; macro metrics and AUPRC; multi-seed runs; a like-for-like
text-only ablation on identical data; a held-out set at realistic class balance.

**Phase 5 — Evaluate what actually matters for deployment.**
False-positive rate on reclaimed in-group language. Adversarial robustness (character substitution,
text-in-image obfuscation, crops). Subgroup breakdowns. Qualitative error analysis with community
input.

**Phase 6 — Infrastructure from day one.**
Experiment tracking, config-driven sweeps, versioned data, seeded runs.

## The transferable lessons

**Data work dominates.** Most of the elapsed time went to acquisition, cleaning, and labeling — and
nearly every failed modeling experiment traced back to dataset size. The instinct to spend the next
week on architecture was, in this project, almost always wrong.

**Constraints produce defensible designs.** No compute forced frozen embeddings, which made the
pipeline fast, cheap, and reproducible. Worth noticing when a constraint improves a design.

**Measure the thing you claim.** The central claim — "multimodal beats text-only" — was never
cleanly isolated, because the baseline used a different corpus. One controlled experiment would have
fixed it.

**Bigger models don't win by default.** `RN50x64` had the most parameters and the widest embedding,
and finished last. Representation quality beat capacity.

**Absence of evaluation is a result.** No error analysis, no fairness testing, no adversarial
testing, no κ. Each of those gaps is information about what the project can and cannot claim.

**Deleting inconvenient data is a methodological error.** Dropping annotator disagreements made the
metrics look better and the work worth less.

---

**Previous:** [`04_evaluation.md`](04_evaluation.md) · **Back to:** [`../README.md`](../README.md)
