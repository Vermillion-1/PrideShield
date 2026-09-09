# PrideShield

**Multimodal detection of anti-LGBTQ+ hate speech in internet memes.**

Memes are a hard case for content moderation: the image is often benign, the overlaid text is
often benign, and the hostility only exists in their *combination*. PrideShield attacks that
problem with a multimodal classifier — OpenAI **CLIP** visual and text embeddings fused into an
MLP head — benchmarked against a **BERT** text-only baseline to quantify how much the visual
modality actually contributes.

The repository contains the complete research pipeline: multi-platform data collection, OCR
extraction and correction, perceptual deduplication, dual-annotator label reconciliation, and
model training with hyperparameter search.

> **Status: archived research code (2024–2025).** This was built as an undergraduate major
> project and reflects the multimodal tooling of that period. It is preserved as-is for
> reproducibility and reference. See **[Future Scope](#future-scope)** for an honest assessment of
> how this would be built differently today.

---

## Table of Contents

- [Motivation](#motivation)
- [Approach](#approach)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Pipeline](#pipeline)
- [Getting Started](#getting-started)
- [Dataset & Responsible Use](#dataset--responsible-use)
- [Limitations](#limitations)
- [Future Scope](#future-scope)
- [Citation](#citation)
- [License](#license)

---

## Motivation

Automated moderation systems are typically strong on text and weak on multimodal content.
Anti-LGBTQ+ harassment in meme form exploits exactly that gap — the meaning is carried by the
interaction between image and caption, so unimodal classifiers systematically under-detect it.

This project asks a narrow, testable question: **does adding the visual modality measurably improve
detection over a strong text-only baseline on this class of content?**

## Approach

**Text-only baseline.** A DistilBERT/BERT-family transformer fine-tuned for binary sequence
classification (attention dropout 0.2, hidden dropout 0.2, HuggingFace `Trainer`), trained and
evaluated on a *public* English hate-speech benchmark (FRENK, LGBT subset). This establishes what
text alone achieves.

**Multimodal model.** For each meme:
1. The image is embedded with a **CLIP** vision encoder.
2. The OCR-extracted caption is embedded with the **CLIP** text encoder (chunked to respect the
   77-token context limit, chunk embeddings averaged).
3. Both embeddings are L2-normalized and **concatenated** into a single feature vector.
4. An **MLP classification head** is trained on the fused vector.

**Class imbalance** is handled with a from-scratch **Focal Loss** implementation
(the `(1 − p_t)^γ` down-weighting of easy examples) and, as an alternative, inverse-frequency
class weighting. Which one is used is selected empirically rather than assumed.

**Hyperparameter search.** The final model is selected by an **Optuna** study (30 trials) jointly
tuning optimizer, learning rate, weight decay, dropout, activation, loss function, and MLP
width/depth — rather than hand-tuned.

**Vision encoders benchmarked:** `ViT-B/32`, `ViT-L/14@336px`, and `RN50x64`.

## Results

**Best configuration — CLIP `ViT-L/14@336px` embeddings → MLP head**, selected by Optuna
(AdamW, lr 0.01, weight decay 0, dropout 0.4, LeakyReLU, Focal Loss):

| Metric | Score |
|---|---|
| Accuracy | **93.94%** |
| Precision | 96.91% |
| Recall | 94.58% |
| F1 | 95.73% |
| ROC-AUC | 0.9030 |

**Vision encoder comparison.** All three CLIP encoders were run through the same downstream MLP
and search procedure:

| Encoder | Embedding dim | Params | Peak accuracy | Relative compute |
|---|---|---|---|---|
| `ViT-B/32` | 512 | ~88M | 89.73% | Low |
| **`ViT-L/14@336px`** | 768 | ~307M | **93.94%** | High |
| `RN50x64` | 1024 | ~336M | 84.85% | Very high |

The larger ViT wins, and — notably — the CNN encoder (`RN50x64`) loses despite having the most
parameters and the widest embedding, suggesting the gain comes from representation quality rather
than capacity. The text-only transformer baseline reached ~86.87%, so the multimodal pipeline
adds roughly 7 points over text alone.

**Read these numbers with the caveats in [Limitations](#limitations).** In particular: the dataset's
class balance is inverted relative to real-world moderation traffic (a majority-class baseline
already scores ~80.9%), precision/recall/F1 are positive-class rather than macro figures, and the
text baseline was trained on a *different, external* corpus — so the ~7-point gap is indicative,
not a controlled ablation of the visual modality.

## Architecture

<p align="center">
  <img src="docs/diagrams/data_flow_diagram.jpg" alt="Data flow diagram" width="720">
</p>

**Classifier head.** CLIP embeddings feed a compact MLP:

```
Linear(embed_dim, 256) → ReLU → Dropout(0.4)
    → Linear(256, 128) → ReLU → Dropout(0.4)
        → Linear(128, 2)
```

trained under either CrossEntropy or Focal Loss, with the choice made by the Optuna study rather
than fixed in advance.

Additional design documentation — class diagram, entity-relationship diagram, and use-case
diagram — is in [`docs/diagrams/`](docs/diagrams/) (editable `.drawio` sources included).

## Repository Structure

```
PrideShield/
├── notebooks/
│   ├── 01_data_collection/     # multi-platform scraping
│   ├── 02_preprocessing/       # OCR, dedup, normalization, annotation merge
│   ├── 03_modeling/            # BERT baseline, CLIP+MLP models
│   └── utils/                  # shared data-wrangling helpers
├── docs/
│   ├── diagrams/               # DFD, class, ER, use-case (+ .drawio sources)
│   └── assets/                 # project introduction video
├── requirements.txt
├── CITATION.cff
├── LICENSE                     # Apache-2.0
└── NOTICE
```

## Pipeline

Notebooks are numbered in execution order within each stage.

### `01_data_collection/`
| Notebook | Purpose |
|---|---|
| `01_reddit_scraper_praw.ipynb` | Keyword-targeted collection from selected subreddits via PRAW |
| `02_reddit_scraper_async_praw.ipynb` | Async (`asyncpraw`) variant for higher-throughput collection |
| `03_4chan_scraper.ipynb` | Thread/post/media collection via 4chan's public JSON API |
| `04_selenium_image_scraper.ipynb` | Selenium-driven image collection for JS-rendered sources |
| `05_google_images_scraper.ipynb` | Query-based supplementary image collection |

### `02_preprocessing/`
| Notebook | Purpose |
|---|---|
| `01_ocr_text_extraction.ipynb` | EasyOCR text extraction from meme images |
| `02_ocr_correction_t5.ipynb` | T5-based correction of OCR errors (garbled tokens, spacing) |
| `03_image_deduplication.ipynb` | Perceptual-hash deduplication (`imagededup`) |
| `04_reverse_image_search_validation.ipynb` | Reverse-image search to validate provenance |
| `05_integrate_supplementary_images.ipynb` | Merge supplementary images into the main corpus |
| `06_merge_annotation_csvs.ipynb` | Consolidate independent annotators' label files |
| `07_text_normalization_spellcheck.ipynb` | Levenshtein/spellchecker normalization of OCR text |

**Annotation protocol:** every meme was labeled independently by **two annotators**; only rows where
both agreed were retained, and disagreements were dropped. This raises label precision but removes
the hardest, most ambiguous examples — see [Limitations](#limitations).

### `03_modeling/`
| Notebook | Purpose |
|---|---|
| `01_bert_text_baseline.ipynb` | BERT text-only baseline on the public FRENK-LGBT benchmark |
| `02_bert_finetuning_v1.ipynb` | Further fine-tuning experiments on the text model |
| `03_clip_mlp_v1.ipynb` | First CLIP-embedding + MLP multimodal model |
| `04_clip_mlp_optuna_final.ipynb` | **Final model** — Optuna search, Focal Loss, encoder comparison |
| `05_multimodal_architecture_experiments.ipynb` | Alternative fusion architectures (incl. a tabular-transformer variant) |

## Getting Started

```bash
git clone https://github.com/Vermillion-1/PrideShield.git
cd PrideShield
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Notebooks were developed in Google Colab and contain Colab-specific paths (`/content/…`,
Drive mounts). Adjust paths for local execution. Scrapers require your own API credentials
(Reddit app credentials for PRAW); never commit them — `.gitignore` covers `praw.ini` and `.env`.

Notebook outputs are intentionally stripped from this repository (see below).

## Dataset & Responsible Use

**The dataset is not distributed in this repository, and the notebook outputs have been cleared.**

This is deliberate. The corpus consists of scraped hate speech targeting a marginalized community.
Publishing it — or the rendered notebook outputs containing it — would redistribute that material
to anyone browsing the repo, and would conflict with the source platforms' terms of service.
The code is public; the harmful content is not.

Researchers with a legitimate use may contact the author to discuss access.

**If you build on this work:** treat the classifier as a *triage aid*, not an adjudicator. False
positives on reclaimed in-group language are a known and serious failure mode for this class of
model, and the cost of silencing the community you intend to protect is not symmetric with the cost
of missing a slur.

## Limitations

Stated plainly, because they bound what the results mean:

- **Class balance is inverted relative to deployment.** The dataset is roughly 80.9% / 19.1%,
  whereas real moderation traffic is overwhelmingly benign. A majority-class baseline scores ~80.9%
  here, so headline accuracy overstates practical utility — minority-class precision/recall are the
  informative metrics.
- **Reported precision/recall/F1 are positive-class, not macro.** Macro-averaged figures are
  materially lower.
- **The text baseline is not a clean ablation.** It was trained on an external public benchmark
  (FRENK-LGBT), not on the meme corpus, so "multimodal beats text-only" is not a controlled
  comparison on identical data.
- **Small evaluation set.** The meme corpus is on the order of ~1.3k labeled examples; differences
  between CLIP encoder variants are point estimates without confidence intervals and may not be
  statistically separable.
- **Annotator disagreements were dropped, not adjudicated**, removing the hardest cases and
  likely inflating all reported metrics relative to real-world difficulty.
- **Single-run results.** No multi-seed variance reporting; no k-fold cross-validation, so the
  encoder-comparison gaps are point estimates.
- **Scope narrowed during the project.** The original objective targeted a text corpus of >100k
  observations and a real-time moderation API; the delivered system is an offline research pipeline
  trained on a substantially smaller corpus. The API/intervention layer was specified but not built.
- **No fairness or adversarial-robustness evaluation** was performed — a real gap for a content
  moderation system.
- **English-only**, and scoped to a narrow slice of platforms and time.

## Future Scope

This is 2024-era multimodal tooling. CLIP-embeddings-plus-a-classifier-head was a reasonable design
then; it is no longer the obvious choice. If this work were restarted today:

**1. Benchmark against modern vision-language models first.**
Instruction-tuned VLMs (LLaVA, Qwen-VL, InternVL, or a frontier multimodal API) can be evaluated
zero-shot or few-shot on this task with no training at all. That is the baseline any new work must
beat, and it may well beat the CLIP+MLP pipeline outright. A modern version of this project should
*start* there and justify any trained model against it.

**2. Replace concatenation with real fusion.**
Concatenating frozen CLIP embeddings is early-fusion in its simplest form. Cross-attention between
modalities, or a VLM that jointly attends over image and text natively, can model the
image-text *interaction* that actually carries the hostility — the exact signal concatenation
struggles to represent.

**3. Exploit VLM reasoning for explainability.**
A VLM can be prompted to *explain why* a meme is hateful, producing a rationale a human moderator
can audit. That is far more useful operationally than a scalar score, and it addresses the
interpretability gap in the current pipeline.

**4. Fix the evaluation methodology.**
The genuinely important upgrades are not architectural: a like-for-like text-only vs. multimodal
ablation on identical data; k-fold cross-validation with confidence intervals; macro-averaged
metrics; multi-seed runs; adjudicated (not dropped) annotator disagreements; and a held-out set
that reflects realistic class balance.

**5. Add fairness and robustness evaluation.**
Specifically: measure false-positive rates on reclaimed in-group language, and test robustness to
adversarial evasion (character substitution, text-in-image obfuscation, crops).

**6. Track experiments properly.**
Weights & Biases or MLflow, with configuration-driven sweeps rather than duplicated notebook cells,
so results are queryable rather than living in cell outputs.

## Citation

If you use this work, please cite it (GitHub's *"Cite this repository"* button reads
[`CITATION.cff`](CITATION.cff)):

```bibtex
@software{singh_prideshield_2025,
  author  = {Singh, Ankush},
  title   = {{PrideShield: Multimodal Detection of Anti-LGBTQ+ Hate Speech in Memes}},
  year    = {2025},
  url     = {https://github.com/Vermillion-1/PrideShield},
  license = {Apache-2.0}
}
```

## Acknowledgements

Developed by **Ankush Singh** (primary author — data pipeline, preprocessing, modeling,
experimentation). Submitted as an undergraduate major project at Jaypee University of Information
Technology; **Arpan Chauhan** and **Shubh Saxena** are credited as project collaborators on the
academic submission.

## License

Licensed under the **Apache License 2.0** — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
You may use, modify, and distribute this work, including commercially, provided you retain
attribution and the license notice.

---

**Contact:** Ankush Singh — ankush.singh1802@gmail.com
