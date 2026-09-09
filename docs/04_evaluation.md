# 05 — Evaluation & Results

*Retrospective documentation. Reports what was measured, and is explicit about what the numbers do
and do not support.*

---

## Headline results

**Best configuration — CLIP `ViT-L/14@336px` → MLP head**
(AdamW, lr 0.01, weight decay 0, dropout 0.4, LeakyReLU, Focal Loss, layers `[512, 256, 128]`):

| Metric | Score |
|---|---|
| Accuracy | **93.94%** |
| Precision | 96.91% |
| Recall | 94.58% |
| F1 | 95.73% |
| ROC-AUC | 0.9030 |

Evaluated on the held-out test split (~198 examples).

## Comparative results

| Model | Modality | Accuracy |
|---|---|---|
| Majority-class baseline | — | ~80.9% |
| Text-only transformer (FRENK) | Text | ~86.87% |
| CLIP `RN50x64` + MLP | Multimodal | 84.85% |
| CLIP `ViT-B/32` + MLP | Multimodal | 89.73% |
| **CLIP `ViT-L/14@336px` + MLP** | **Multimodal** | **93.94%** |

## How to read these numbers honestly

### The majority-class baseline is the number that matters

The dataset is **80.9% positive**. A model that predicts "hateful" unconditionally scores **80.9%**.

So the real achievement is not "93.94% accuracy" — it is **+13 points over trivial**. And note that
`RN50x64` at 84.85% is only ~4 points above a constant predictor, which reframes it from
"reasonable" to "barely working."

**Any accuracy figure from this project should be quoted alongside the 80.9% floor.** Quoted alone,
it is misleading.

### Precision, recall and F1 are positive-class, not macro

These were computed with `average="binary"` — i.e. **for the majority (hateful) class only**.

Because the positive class holds 80.9% of the data, positive-class metrics are the *easy* metrics.
Macro-averaged figures — which weight the 19.1% minority class equally — are **materially lower**.

The 96.91% precision figure in particular describes performance on the abundant class. It says
little about how the model handles the minority class, which is where the difficult, ambiguous
content lives.

**Macro metrics should have been the headline.** They weren't, and that is a reporting choice that
flatters the result.

### ROC-AUC 0.9030 is the most honest single number

AUC is threshold-independent and less sensitive to class imbalance than accuracy. **0.903 is a
genuinely good discrimination score** and is the figure that best supports "this model learned
something real."

That said: on an imbalanced problem, **AUPRC** (area under the precision–recall curve) is generally
more informative than ROC-AUC, because ROC curves can look optimistic when negatives dominate the
comparison space. We did not compute AUPRC.

### The multimodal gain is not a controlled ablation

The text baseline (~86.87%) and the multimodal model (93.94%) differ in **two** ways: modality *and*
training corpus (FRENK vs. meme captions). The ~7-point gap therefore confounds "adding vision
helps" with "these datasets differ."

The claim the evidence supports is: *a multimodal model trained on this meme corpus outperforms a
text model trained on a general LGBT hate-speech corpus.* The claim it does **not** support is:
*adding the visual modality yields +7 points.*

Fixing this requires only training the text baseline on meme OCR captions — cheap, and not done.

## What was *not* measured

Listed explicitly, because absence of evaluation is itself a result:

| Not measured | Why it matters |
|---|---|
| **Macro-averaged P/R/F1** | The reported metrics favor the majority class |
| **AUPRC** | More informative than ROC-AUC under imbalance |
| **Confidence intervals** | 198 test examples → ±~3–4 points; small gaps may be noise |
| **Multi-seed variance** | Single run per configuration; no stability estimate |
| **k-fold cross-validation** | Would have been nearly free on cached embeddings |
| **Confusion matrix analysis** | Which classes fail, and how, was never examined |
| **Qualitative error analysis** | *Which memes* the model gets wrong — the most informative analysis available, never done |
| **Reclaimed-language false-positive rate** | The single most consequential deployment failure mode |
| **Adversarial robustness** | Character substitution, obfuscation, crops — all untested |
| **Fairness / subgroup analysis** | No breakdown across content types or communities |
| **Ablation of the OCR correction stack** | Its contribution to final accuracy is unknown |
| **Ablation of fusion strategy** | Concatenation vs. alternatives never compared |

## The evaluation-set size problem

The test split is **~198 examples**. Consequences:

- One misclassification ≈ **0.5 percentage points**.
- The 95% confidence interval around 93.94% is roughly **±3–4 points**.
- The `ViT-B/32` (89.73%) vs `ViT-L/14` (93.94%) gap is ~4 points — **plausibly real but not
  statistically established** from a single split.
- The `RN50x64` (84.85%) result is far enough below the others to be trustworthy as a ranking.

**k-fold cross-validation was the correct protocol** for a dataset this size and would have cost
minutes on precomputed embeddings. Its absence is the clearest evaluation weakness.

## Test-case results in the project report — do not cite these

The project report includes figures like "98% capture on explicit insults" and "95% on implicit
statements." These come from **hand-constructed test sentences**, not held-out evaluation data.

They are demonstrations, not measurements: the inputs were written by the same people who built the
system, with no sampling protocol and no independent construction. **They should not be reported as
model performance metrics.** They are excluded from the README for this reason.

## Metric provenance and one discrepancy

Numbers in this document come from the **project report** (`G101.pdf`), the submitted and defended
artifact.

One documented inconsistency: the report gives the text baseline at **~86.87%**, while a surviving
notebook output shows **77%** on a held-out split. These are most likely different runs at different
stages. The report takes precedence, but the discrepancy is recorded rather than smoothed over.

More broadly: **results were never tracked in a proper experiment log.** Metrics live in report
tables and notebook cell outputs, which is why reconstructing them required cross-referencing
multiple sources. An experiment tracker (W&B, MLflow) or even a disciplined `results.csv` would have
made every number in this document trivially verifiable.

## What the results actually support

Stated conservatively:

✅ **Supported:**
- A CLIP-embedding multimodal classifier substantially outperforms trivial baselines on this corpus
  (93.94% vs 80.9%).
- Vision-transformer encoders outperform the CNN encoder for this task, and the largest ViT performs
  best.
- ROC-AUC 0.903 indicates genuine discriminative capability, not a degenerate classifier.
- Automated hyperparameter search over loss functions surfaced a real, encoder-dependent
  interaction.

❌ **Not supported:**
- A precise quantification of the visual modality's contribution (confounded baseline).
- Deployment readiness (untested on realistic class balance, no fairness or robustness evaluation).
- Statistical separation of the closer encoder comparisons.
- Any claim about performance on ambiguous or contested content — those examples were removed from
  the dataset before evaluation.

---

**Previous:** [`03_modeling.md`](03_modeling.md) · **Next:** [`05_future_work.md`](05_future_work.md)
