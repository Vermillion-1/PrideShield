# 02 — Preprocessing & Data Cleaning

Turning ~1,700 raw scraped images into a usable multimodal training table.

---

## The four problems

| Problem | Solution | Notebook |
|---|---|---|
| Same meme appears across every source | Perceptual-hash deduplication | `03_image_deduplication` |
| Text is *inside* the image, not in metadata | EasyOCR extraction | `01_ocr_text_extraction` |
| OCR output frequently garbled | T5 correction + Levenshtein repair | `02_ocr_correction_t5`, `07_text_normalization_spellcheck` |
| Provenance of reposts unclear | Selenium reverse image search | `04_reverse_image_search_validation` |

## Deduplication

**Tool:** `imagededup`, perceptual hashing (pHash).

**Why perceptual, not exact:** memes are re-encoded, re-compressed, watermarked and cropped as they
spread. MD5/SHA see these as entirely different files. pHash produces similar hashes for
perceptually similar images, so near-duplicates collapse.

**Why it's a correctness requirement:** duplicates landing on both sides of a train/test split leak
the test set — the model memorizes rather than generalizes, and reported accuracy becomes fiction.
This was not housekeeping.

## OCR extraction

**Tool:** EasyOCR.

Meme text is the harder half of the multimodal signal and genuinely difficult to extract: stylized
and heavily-outlined fonts, low contrast against busy backgrounds, text spread across panels with no
reading order, and text rendered *as part of* the image rather than overlaid.

Observed failure modes: dropped characters, merged words, spurious characters from background
texture, scrambled reading order on multi-panel memes.

**Why it mattered so much:** the caption embedding is half the classifier input, and **CLIP embeds
nonsense confidently**. Garbled OCR doesn't add noise — it produces a confidently wrong text
embedding that poisons the fused representation. This made OCR quality one of the highest-leverage
variables in the pipeline.

## OCR correction — two stages

### Stage 1: T5 sequence-to-sequence correction

A T5 model applied as a corrector — garbled OCR in, plausible text out. Handles **structural**
damage (merged words, lost spacing, dropped characters) that spans token boundaries and so cannot
be fixed by token-level spellchecking.

### Stage 2: Levenshtein / spellchecker repair

Token-level repair of residual character errors using edit distance (`python-Levenshtein`) and
`pyspellchecker`.

```
lev(i,j) = max(i,j)                             if min(i,j) = 0
         = min( lev(i-1, j)   + 1,
                lev(i,   j-1) + 1,
                lev(i-1, j-1) + 1[ai != bj] )   otherwise
```

**Second motivation:** hate speech routinely uses deliberate misspellings and character substitution
to evade keyword filters. Edit-distance normalization partially recovers the intended token.

## Provenance validation

Selenium-driven reverse image search identifies reposts with altered captions and images whose
scraped context differs from their origin. Applied selectively — too slow to run exhaustively.

## Corpus integration

`05_integrate_supplementary_images` merges later collection batches, **re-running deduplication at
merge time** so top-up scrapes don't reintroduce duplicates the earlier pass removed.

## Label reconciliation

`06_merge_annotation_csvs` joins the two independent label files on row index and keeps only
agreeing items -> **1,667 -> 1,320**.

## Output

| Column | Content |
|---|---|
| `Index` | Row identifier |
| `Filename` | Image reference |
| `Extracted Text` | OCR text after T5 + Levenshtein correction |
| `Annotator_1` / `Annotator_2` | Independent labels |

---

## Notes & Caveats

<sub>

**The correction stack was never ablated.** There is no measurement of raw OCR vs. T5-corrected vs.
fully-normalized text through to final accuracy. It was believed to help based on visual inspection
of corrected strings — probably correct, but unevidenced, and a cheap experiment on cached
embeddings.

**OCR quality was never measured.** No character error rate against hand-transcribed ground truth,
and no benchmarking against alternatives (PaddleOCR, Tesseract with preprocessing). EasyOCR was
adopted because it worked, not because it was compared.

**Adversarial robustness is design intent, not demonstrated capability.** The edit-distance stage
was never tested against actual adversarial inputs.

**Deduplication is not exactly reproducible** — the pHash distance threshold and the number of pairs
merged versus kept were not logged.

**Dropping contested labels** removed the hardest examples; see [`01_dataset.md`](01_dataset.md).

</sub>

---

**Previous:** [`01_dataset.md`](01_dataset.md) · **Next:** [`03_modeling.md`](03_modeling.md)
