# 04 — Modeling

*Retrospective documentation. Covers the text baseline, the multimodal architecture, the encoder
comparison, and the variants that were tried and abandoned.*

---

## Design constraint: frozen encoders

Free-tier Colab could not fine-tune a 307M-parameter vision transformer. The architecture was
therefore fixed early:

> **Frozen CLIP encoders as feature extractors + a small trainable MLP head.**

This was a compute decision before it was a modeling decision. It has real consequences — the
encoders never adapt to meme-specific visual style or the peculiarities of OCR text — but it also
brought genuine advantages worth acknowledging: embeddings are computed **once** and cached, after
which every experiment trains in seconds on CPU. That is what made a 30-trial hyperparameter search
and a three-encoder comparison affordable at all. Under the constraints, it was the right call.

## Text-only baseline

**Purpose:** establish what text alone achieves, so the multimodal model has something to beat.

| Property | Value |
|---|---|
| Model | DistilBERT / BERT-family (`AutoModelForSequenceClassification`, `num_labels=2`) |
| Attention dropout | 0.2 |
| Hidden dropout | 0.2 |
| Training | HuggingFace `Trainer` |
| Data | **FRENK hate-speech benchmark, LGBT subset** (public, English) |
| Reported accuracy | **~86.87%** (project report) |

⚠️ **Two caveats that matter:**

1. **Model identity discrepancy.** The project report specifies `distilbert-base-uncased`. The
   surviving notebook code instantiates `bert-base-uncased`, with DistilBERT present but commented
   out. These were likely different runs at different times. The safe description is
   "a DistilBERT/BERT-family baseline" — don't over-specify.
2. **Metric discrepancy.** The report states ~86.87%; a surviving notebook output shows 77% on a
   held-out split. Most likely an earlier run versus the final one. The report is the submitted,
   defended artifact, so it takes precedence — but the discrepancy is real and is recorded here
   rather than hidden.

**The deeper problem:** this baseline was trained on FRENK, *not* on meme OCR captions. So the
text-vs-multimodal comparison varies **both** modality and training corpus. It is not a controlled
ablation, and the ~7-point multimodal gain is indicative rather than isolated. Training the baseline
on the meme captions would have cost almost nothing and would have made the central claim clean.
This is the project's most significant methodological miss.

## Multimodal architecture

For each meme:

```
   image ──► CLIP vision encoder (frozen) ──► image embedding ──┐
                                                                 ├─► L2-norm ─► concat ─► MLP ─► {hateful, not}
   OCR text ► CLIP text encoder (frozen) ──► text embedding ────┘
```

**Steps:**

1. **Image embedding** — frozen CLIP vision encoder.
2. **Text embedding** — frozen CLIP text encoder over the corrected OCR caption.
3. **77-token workaround.** CLIP's text encoder has a hard 77-token context limit; longer captions
   are chunked, each chunk embedded, and the chunk embeddings **averaged**. This is informal and
   lossy — averaging discards word order and dilutes salient content in long captions.
4. **L2 normalization** of both vectors, then **concatenation**.
5. **MLP classification head** on the fused vector.

### Classifier head

```
Linear(embed_dim, 256) → ReLU → Dropout(0.4)
    → Linear(256, 128) → ReLU → Dropout(0.4)
        → Linear(128, 2)
```

Deliberately small — with ~924 training examples, a larger head overfits immediately.

## Encoder comparison

All three CLIP encoders were run through **identical** downstream conditions (same head, same
search procedure, same splits), which is what makes the comparison meaningful.

| Encoder | Type | Embed dim | Params | Peak accuracy |
|---|---|---|---|---|
| `ViT-B/32` | Vision Transformer, 32px patches | 512 | ~88M | 89.73% |
| **`ViT-L/14@336px`** | Vision Transformer, 14px patches, 336px input | 768 | ~307M | **93.94%** |
| `RN50x64` | Scaled ResNet (CNN) | 1024 | ~336M | 84.85% |

### The interesting result: the CNN lost despite being the largest

`RN50x64` has the **most parameters (~336M)** and the **widest embedding (1024-d)**, and finished
**last** — nearly 5 points below `ViT-B/32`, which has roughly a quarter of the parameters and half
the embedding width.

Capacity was not the binding factor; **representation quality** was. A plausible reading: this task
requires relating overlaid text semantics to image semantics, and ViT's global self-attention
captures that relational structure in a way convolutional inductive bias does not. CLIP's ViT and
ResNet variants also differ in training dynamics beyond architecture, so this is a reasonable
interpretation rather than a proven mechanism.

**Caveat:** on a 198-example test set these gaps are point estimates without confidence intervals.
The ViT-L vs ViT-B gap (~4 points) is probably real; smaller gaps may not be separable.

### Patch size and resolution

`ViT-L/14@336px` wins on three axes simultaneously — finer patches (14 vs 32), higher input
resolution (336 vs 224), more parameters. The design does not isolate which contributes most. Finer
patches at higher resolution should help with small overlaid text, which is a sensible hypothesis
for *why* it won, but the experiment doesn't demonstrate it.

## Class imbalance handling

With 80.9% / 19.1% balance, an untreated model can score ~80.9% by always predicting the majority
class. Two strategies were implemented:

**1. Focal Loss** — written from scratch:

```
FL(p_t) = -α (1 - p_t)^γ log(p_t)
```

The `(1 - p_t)^γ` term down-weights already-easy examples, concentrating gradient on hard ones.

**2. Class-weighted CrossEntropy** — inverse-frequency weighting.

**Neither was assumed correct.** The loss function was a *search parameter*, and Optuna chose:

| Encoder | Loss selected |
|---|---|
| ViT-L/14@336px | **Focal Loss** |
| ViT-B/32 | **Class-weighted CrossEntropy** |

That split is a genuine finding: the optimal imbalance strategy depended on the encoder, which is
not something we would have discovered by fixing the loss in advance.

## Hyperparameter search

**Optuna, 30 trials**, jointly tuning:

| Parameter | Search space |
|---|---|
| Optimizer | Adam / AdamW / SGD |
| Learning rate | continuous |
| Weight decay | continuous |
| Dropout | continuous |
| Activation | ReLU / LeakyReLU / GELU |
| **Loss function** | Focal Loss / CrossEntropy |
| Layer configuration | width and depth |

**Winning configuration (ViT-L/14@336px):** AdamW, lr 0.01, weight decay 0, dropout 0.4,
LeakyReLU, Focal Loss, layers `[512, 256, 128]`.

**In hindsight:** 30 trials is thin for a 7-dimensional space, and the search optimized a single
validation split — with 198 validation examples, some of the "best" configuration is fitted to
validation noise. Nested cross-validation would have been the correct protocol and was affordable
on cached embeddings.

## What didn't work

| Attempt | Outcome | Why (in hindsight) |
|---|---|---|
| **Transformer over tabular features** — a transformer encoder over the concatenated embedding vector | Abandoned; no gain over plain MLP | ~924 training examples cannot support the extra capacity; there is no sequential structure in a concatenated embedding for attention to exploit |
| **MLP + BatchNorm** | No consistent improvement | Small batches on a small dataset make BatchNorm statistics noisy |
| **MLP + L2 + early stopping** | No improvement over Optuna-selected dropout | Dropout at 0.4 was already doing the regularization work |
| **Deeper MLP heads** | Overfit | Dataset size, again |
| **RN50x64 encoder** | Worst performer despite largest size | See above — representation quality over capacity |

**The consistent theme:** every attempt to add capacity failed, and the binding constraint was
always dataset size. That is the correct diagnosis, and it points at data — not architecture — as
where effort should have gone.

## The fusion limitation

Concatenating two frozen embeddings is **early fusion in its crudest form**. The MLP sees two
independent vectors glued together and must infer their relationship from scratch, with no mechanism
for the modalities to attend to each other.

For memes this is a real handicap: hostility lives in the *interaction* between image and caption —
a benign image paired with a benign phrase producing a hateful whole. Concatenation cannot
represent "this text, about this image."

We knew this at the time. Cross-attention fusion was out of reach on free-tier compute, and this is
the single clearest architectural upgrade for any continuation (see
[`05_future_work.md`](05_future_work.md)).

---

**Previous:** [`02_preprocessing.md`](02_preprocessing.md) · **Next:** [`04_evaluation.md`](04_evaluation.md)
