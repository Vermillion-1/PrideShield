# 02 — Preprocessing & Data Cleaning

*Retrospective documentation. This stage consumed more effort than modeling and had a larger effect
on final performance than most architectural choices.*

---

## Overview

Raw scraped memes are not usable as-is. Four distinct problems had to be solved:

| Problem | Solution | Notebook |
|---|---|---|
| The same meme appears across every source | Perceptual-hash deduplication | `03_image_deduplication.ipynb` |
| Text is *inside* the image, not in metadata | EasyOCR extraction | `01_ocr_text_extraction.ipynb` |
| OCR output is frequently garbled | T5 sequence correction + Levenshtein repair | `02_ocr_correction_t5.ipynb`, `07_text_normalization_spellcheck.ipynb` |
| Provenance of reposted/edited images is unclear | Selenium reverse image search | `04_reverse_image_search_validation.ipynb` |

## Deduplication

**Tool:** `imagededup`, perceptual hashing (pHash).

**Why perceptual rather than exact hashing:** identical memes are re-encoded, re-compressed,
watermarked, and cropped as they spread. An MD5/SHA hash sees these as entirely different files.
Perceptual hashing produces similar hashes for perceptually similar images, so near-duplicates
collapse together.

**Why this mattered more than it looks:** duplicates spanning a train/test boundary silently leak
the test set. The model memorizes rather than generalizes, and reported accuracy becomes fiction.
Deduplication was not housekeeping — it was a correctness requirement.

**In hindsight:** we did not record the pHash distance threshold used, nor how many near-duplicate
pairs were merged versus kept. Those numbers should have been logged; the deduplication is not
precisely reproducible from the notebook alone.

## OCR extraction

**Tool:** EasyOCR.

Meme text is the harder half of the multimodal signal, and it is genuinely difficult to extract:

- Stylized, decorative, or heavily-outlined fonts
- Low contrast against busy backgrounds
- Text overlaid across multiple image regions with no reading order
- Impact-font all-caps with heavy stroke effects
- Text rendered *as part of* the image content rather than as an overlay

EasyOCR handled the common cases and failed on the rest. Failure modes we observed: dropped
characters, merged words, spurious characters from background texture, and reading-order scrambling
on multi-panel memes.

**Why this was critical:** the caption embedding is half the input to the classifier. Garbled OCR
does not merely add noise — it produces a *confidently wrong* text embedding, because CLIP will
happily embed nonsense. Poor OCR poisons the fused representation.

**In hindsight:** we should have benchmarked EasyOCR against alternatives (PaddleOCR, Tesseract with
preprocessing, or a cloud OCR free tier) and measured character error rate on a hand-transcribed
sample. We had no OCR quality metric at all — we corrected downstream and assumed it helped.

## OCR correction — a two-stage repair layer

Because raw OCR was unusable, we built a correction stack on top.

### Stage 1 — T5 sequence correction (`02_ocr_correction_t5.ipynb`)

A T5 model (HuggingFace `transformers`) applied as a sequence-to-sequence corrector: garbled OCR
in, plausible text out. This handles *structural* damage — merged words, missing spacing, dropped
characters — that token-level spellchecking cannot.

### Stage 2 — Levenshtein / spellchecker repair (`07_text_normalization_spellcheck.ipynb`)

Token-level repair using edit distance (`python-Levenshtein`) and `pyspellchecker` for residual
character-level errors.

**Levenshtein distance** — the minimum single-character insertions, deletions, or substitutions to
transform one string into another:

```
lev(i,j) = max(i,j)                             if min(i,j) = 0
         = min( lev(i-1, j)   + 1,
                lev(i,   j-1) + 1,
                lev(i-1, j-1) + 1[aᵢ ≠ bⱼ] )    otherwise
```

**A second motivation:** adversarial evasion. Hate speech online routinely uses deliberate
misspellings, character substitution, and symbol swaps to defeat keyword filters. Edit-distance
normalization partially recovers the intended token. We never tested this systematically against
adversarial inputs, so it remains an untested design intent rather than a demonstrated capability.

**In hindsight — the biggest gap in this stage:** we never measured whether the correction stack
actually improved end-task performance. There is no ablation of *raw OCR vs. T5-corrected vs.
fully-normalized* text through to final accuracy. We believed it helped based on visual inspection
of corrected strings. That belief is probably right, but it is unevidenced, and it would have been
a cheap experiment on frozen embeddings.

## Reverse image search validation

**Tool:** Selenium-driven reverse image search (`04_reverse_image_search_validation.ipynb`).

Used to check provenance where it mattered — identifying reposts with altered captions, and images
whose original context differed from their scraped context. Applied selectively rather than
exhaustively; it was too slow to run across the full corpus.

## Supplementary image integration

`05_integrate_supplementary_images.ipynb` merges later collection batches into the main corpus,
re-running deduplication so that top-up scrapes don't reintroduce duplicates.

## Label reconciliation

`06_merge_annotation_csvs.ipynb` joins the two independent label files on row index and keeps only
items where both labels agree, producing the final training table. See
[`01_dataset.md`](01_dataset.md) for what this costs the dataset.

## Final preprocessing output

The pipeline's output is a table with, per meme:

| Column | Content |
|---|---|
| `Index` | Row identifier |
| `Filename` | Image file reference |
| `Extracted Text` | OCR text after T5 + Levenshtein correction |
| `Annotator_1` | First annotator's label |
| `Annotator_2` | Second annotator's label |

1,667 rows at this stage, reduced to 1,320 after disagreement removal.

## What we would do differently

1. **Measure OCR quality.** Hand-transcribe ~100 memes, compute character error rate, and use it to
   choose between OCR engines instead of defaulting to the first one that worked.
2. **Ablate the correction stack** end-to-end (raw / T5 / T5+Levenshtein → final accuracy). Cheap
   on frozen embeddings; we simply didn't think to.
3. **Log deduplication parameters and outcomes** so the step is reproducible.
4. **Test adversarial robustness explicitly** rather than assuming normalization confers it.
5. **Use a VLM for OCR.** Modern vision-language models read stylized meme text substantially better
   than 2024-era OCR *and* can be prompted to preserve reading order across panels — this single
   substitution would likely have removed the entire correction stack.

---

**Previous:** [`01_dataset.md`](01_dataset.md) · **Next:** [`03_modeling.md`](03_modeling.md)
