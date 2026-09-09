# 03 — Modeling

Text baseline, multimodal architecture, encoder comparison, and the variants that were rejected.

---

## Design constraint: frozen encoders

Free-tier Colab could not fine-tune a 307M-parameter vision transformer, fixing the architecture
early:

> **Frozen CLIP encoders as feature extractors + a small trainable MLP head.**

A compute decision before a modeling one — but it had a real upside. Embeddings are computed **once**
and cached, after which every experiment trains in seconds on CPU. That is what made a 30-trial
hyperparameter search across three encoders affordable at all.

## Text-only baseline

**Purpose:** establish what text alone achieves, so the multimodal model has a meaningful target.

| Property | Value |
|---|---|
| Model | DistilBERT / BERT-family (`AutoModelForSequenceClassification`, 2 labels) |
| Dropout | attention 0.2, hidden 0.2 |
| Training | HuggingFace `Trainer` |
| Data | FRENK hate-speech benchmark, LGBT subset (public) |
| **Accuracy** | **~86.87%** |

## Multimodal architecture

```
image ----> CLIP vision encoder (frozen) ----> L2-norm --+
                                                          +--> concat --> MLP --> {hateful, not}
OCR text --> CLIP text encoder   (frozen) ----> L2-norm --+
```

1. **Image embedding** — frozen CLIP vision encoder.
2. **Text embedding** — frozen CLIP text encoder over the corrected OCR caption.
3. **77-token workaround** — CLIP's text encoder has a hard 77-token limit; longer captions are
   chunked and the chunk embeddings averaged.
4. **L2 normalisation** of both vectors, then **concatenation** — normalising first prevents the
   larger-magnitude modality from dominating.
5. **MLP head** on the fused vector.

### Classification head

```
Linear(embed_dim, 256) -> ReLU -> Dropout(0.4)
    -> Linear(256, 128) -> ReLU -> Dropout(0.4)
        -> Linear(128, 2)
```

Deliberately small — with ~924 training examples a larger head overfits immediately. Extracted to
[`../src/models.py`](../src/models.py).

## Encoder comparison

All three run through **identical** downstream conditions — same head, same search, same splits.

| Encoder | Type | Embed dim | Params | Accuracy |
|---|---|---|---|---|
| `ViT-B/32` | ViT, 32px patches | 512 | ~88M | 89.73% |
| **`ViT-L/14@336px`** | ViT, 14px patches, 336px input | 768 | ~307M | **93.94%** |
| `RN50x64` | Scaled ResNet (CNN) | 1024 | ~336M | 84.85% |

### Finding: the CNN lost despite being the largest

`RN50x64` has the **most parameters** and the **widest embedding**, and finished **last** — ~5 points
below `ViT-B/32`, which has roughly a quarter of its parameters.

Capacity was not the binding factor; **representation quality** was. A plausible reading: the task
requires relating overlaid text semantics to image semantics, and ViT's global self-attention
captures that relational structure in a way convolutional inductive bias does not. CLIP's ViT and
ResNet variants also differ in training dynamics beyond architecture, so this is an interpretation
rather than a proven mechanism.

## Class imbalance

At 80.9%/19.1%, an untreated model reaches ~80.9% by always predicting the majority class. Two
strategies, both in [`../src/losses.py`](../src/losses.py):

**Focal Loss**, implemented from scratch:

```
FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
```

The `(1 - p_t)^gamma` term down-weights easy examples, concentrating gradient on hard ones.

**Class-weighted CrossEntropy** — inverse-frequency weighting.

**Neither was assumed correct.** The loss function was made a *search parameter*:

| Encoder | Loss selected by Optuna |
|---|---|
| `ViT-L/14@336px` | **Focal Loss** |
| `ViT-B/32` | **Class-weighted CrossEntropy** |

That split is a genuine finding — the right imbalance strategy depended on the encoder, which fixing
the loss in advance would have concealed.

## Hyperparameter search

**Optuna, 30 trials**, jointly tuning optimizer (Adam/AdamW/SGD), learning rate, weight decay,
dropout, activation (ReLU/LeakyReLU/GELU), **loss function**, and layer configuration.

**Winning configuration:** AdamW, lr 0.01, weight decay 0, dropout 0.4, LeakyReLU, Focal Loss,
layers `[512, 256, 128]`.

## Rejected variants

| Attempt | Outcome | Why |
|---|---|---|
| **Transformer over tabular features** | No gain over plain MLP | ~924 examples cannot support the capacity; a concatenated embedding has no sequential structure for attention to exploit |
| **MLP + BatchNorm** | No consistent improvement | Small batches on a small dataset make BatchNorm statistics noisy |
| **MLP + L2 + early stopping** | No improvement | Dropout at 0.4 was already doing the regularization work |
| **Deeper MLP heads** | Overfit | Dataset size |

**The consistent theme:** every attempt to add capacity failed, and the binding constraint was always
dataset size — which locates the problem in the data, not the architecture.

---

## Notes & Caveats

<sub>

**The baseline is not a clean ablation.** It was trained on FRENK, not on meme OCR captions, so the
text-vs-multimodal comparison varies both modality and training corpus. The ~7-point gain is
indicative, not isolated. This is the project's most significant methodological miss and the fix was
cheap.

**Model identity discrepancy.** The project report specifies `distilbert-base-uncased`; surviving
notebook code instantiates `bert-base-uncased` with DistilBERT commented out — likely different runs.
"DistilBERT/BERT-family" is the safe description. Similarly, the report gives the baseline at
~86.87% while a notebook output shows 77%; the report is the submitted artifact and takes precedence.

**Search protocol.** 30 trials is thin for a 7-dimensional space, and the search optimized a single
198-example validation split — so some of the "best" configuration is fitted to validation noise.
Nested cross-validation was the correct protocol and was affordable on cached embeddings.

**Encoder comparison confounds three variables.** `ViT-L/14@336px` wins on patch size, input
resolution *and* parameter count simultaneously; the design does not isolate which contributes.

**Fusion.** Concatenating frozen embeddings is early fusion in its crudest form and cannot model
interaction between modalities — precisely where meme hostility lives. Cross-attention was out of
reach on free-tier compute; see [`05_future_work.md`](05_future_work.md).

**77-token workaround.** Chunking and averaging discards word order and dilutes salient content in
long captions.

</sub>

---

**Previous:** [`02_preprocessing.md`](02_preprocessing.md) · **Next:** [`04_evaluation.md`](04_evaluation.md)
