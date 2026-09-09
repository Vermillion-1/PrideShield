# 04 — Evaluation & Results

What was measured, and how to read it.

---

## Headline

**CLIP `ViT-L/14@336px` -> MLP head** (AdamW, lr 0.01, weight decay 0, dropout 0.4, LeakyReLU,
Focal Loss, layers `[512, 256, 128]`), evaluated on the held-out test split (~198 examples):

| Metric | Score |
|---|---|
| **Accuracy** | **93.94%** |
| Precision | 96.91% |
| Recall | 94.58% |
| F1 | 95.73% |
| ROC-AUC | 0.9030 |

## Comparative

| Model | Modality | Accuracy |
|---|---|---|
| Majority-class baseline | — | ~80.9% |
| CLIP `RN50x64` + MLP | Multimodal | 84.85% |
| Text-only transformer (FRENK) | Text | ~86.87% |
| CLIP `ViT-B/32` + MLP | Multimodal | 89.73% |
| **CLIP `ViT-L/14@336px` + MLP** | **Multimodal** | **93.94%** |

![Results](figures/results.png)

## Reading the numbers

**Against the floor, not zero.** The dataset is 80.9% positive, so a constant predictor scores
80.9%. The achievement is **+13 points over trivial**, not "93.94%". By the same measure `RN50x64`
at 84.85% is only ~4 points above a constant predictor — which reframes it from "reasonable" to
"barely working". **Any accuracy figure from this project should be quoted alongside the 80.9%
floor.**

**ROC-AUC 0.9030 is the most honest single number.** Threshold-independent and less sensitive to
class imbalance than accuracy, it is the figure that best supports "this model learned something
real".

**The multimodal gain is directional, not isolated.** The text baseline and the multimodal model
differ in modality *and* training corpus, so the ~7-point gap confounds the two. What the evidence
supports: *a multimodal model trained on this meme corpus outperforms a text model trained on a
general LGBT hate-speech corpus.* What it does not support: *adding vision yields +7 points.*

## What the results support

**Supported:**
- The classifier substantially outperforms trivial baselines (93.94% vs 80.9%).
- ViT encoders outperform the CNN encoder; the largest ViT performs best.
- ROC-AUC 0.903 indicates genuine discriminative capability, not a degenerate classifier.
- Searching over loss functions surfaced a real, encoder-dependent interaction.

**Not supported:**
- A precise quantification of the visual modality's contribution.
- Deployment readiness — untested on realistic class balance, no fairness or robustness evaluation.
- Statistical separation of the closer encoder comparisons.
- Any claim about ambiguous or contested content — those examples were removed before evaluation.

---

## Notes & Caveats

<sub>

**Metric choice.** Precision/recall/F1 were computed with `average="binary"` — i.e. for the majority
(hateful) class only, which is the easy class at 80.9% prevalence. Macro-averaged figures are
materially lower and should have been the headline. AUPRC is generally more informative than ROC-AUC
under imbalance and was not computed.

**Evaluation-set size.** ~198 test examples means one misclassification is ~0.5 points and the 95%
confidence interval around 93.94% is roughly ±3–4 points. The `ViT-B/32` vs `ViT-L/14` gap (~4
points) is plausibly real but not statistically established from a single split; the `RN50x64` gap
is large enough to trust as a ranking. k-fold cross-validation was the correct protocol and would
have cost minutes on cached embeddings.

**Not measured.** Macro P/R/F1; AUPRC; confidence intervals; multi-seed variance; k-fold CV;
confusion-matrix and qualitative error analysis; false-positive rate on reclaimed in-group language
(the most consequential deployment failure mode); adversarial robustness; calibration;
fairness/subgroup breakdowns; ablation of the OCR-correction stack; ablation of the fusion strategy.
Absence of evaluation is itself a result — it bounds what can be claimed.

**Test-case figures in the project report** ("98% capture on explicit insults", "95% on implicit")
come from hand-constructed sentences written by the project team, not held-out data. They are
demonstrations, not measurements, and are excluded from the README for that reason.

**Provenance.** Figures come from the project report (the submitted, defended artifact). Results were
never tracked in an experiment logger, so reconstructing them required cross-referencing the report
against surviving notebook outputs — which is also why one discrepancy persists (text baseline
~86.87% in the report vs 77% in a notebook output).

</sub>

---

**Previous:** [`03_modeling.md`](03_modeling.md) · **Next:** [`05_future_work.md`](05_future_work.md)
