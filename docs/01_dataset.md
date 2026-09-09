# 01 — Dataset Construction

*Retrospective documentation, written after the fact. This project ran July 2024 – May 2025;
this document reconstructs what was built and why, including decisions that in hindsight were
wrong.*

---

## Why we had to build a dataset at all

There was no publicly available corpus of anti-LGBTQ+ memes with usable labels. The adjacent
resources each fell short in a specific way:

- **Hateful Memes Challenge (Meta)** — the obvious reference dataset, but it is *synthetically
  constructed* and covers hate broadly rather than the LGBTQ+-directed subset, with its own
  distributional quirks.
- **General hate-speech corpora** (Davidson, FRENK, HateXplain) — text-only. They cannot support a
  multimodal task at all, though we did use FRENK's LGBT subset for the text baseline.
- **Platform moderation datasets** — not publicly accessible without institutional partnerships.

So the dataset became the project's first and largest piece of work. In hindsight this is the part
we most underestimated: **the majority of elapsed project time went to acquiring and labeling data,
not to modeling.**

## The budget constraint

This is the context for every decision below: **there was no funding.** No paid API tier, no
annotation vendor, no compute grant.

Concretely this ruled out:

| Wanted | Blocked by | What we did instead |
|---|---|---|
| Twitter/X API for meme collection | Paid-only after 2023 API changes | Dropped X entirely; the `twint` workaround was explored and abandoned |
| Bulk image-search APIs | No free tier at useful volume | Selenium browser automation against DDG/Bing/Google Images |
| Commercial annotation (MTurk, Scale) | Cost per label | Hand-annotated by two project members |
| Dedicated GPU compute | No grant | Google Colab free tier → frozen CLIP embeddings, small trainable head |

**In retrospect:** these constraints were not purely limiting. Being forced onto frozen embeddings
plus a small head produced a pipeline that trains in seconds and is trivially reproducible. The
harm was to *dataset scale*, and scale is what most bounds the results.

## Sources

| Source | Access method | Notebook | Notes |
|---|---|---|---|
| Reddit | PRAW (free tier), keyword + subreddit targeting | `01_reddit_scraper_praw.ipynb` | Rate-limited; ran across multiple sessions |
| Reddit (async) | `asyncpraw` | `02_reddit_scraper_async_praw.ipynb` | Higher throughput variant of the above |
| 4chan | Public JSON API (`/lgbt/` board) | `03_4chan_scraper.ipynb` | No auth required; high hostile-content density |
| DuckDuckGo / Bing Images | Selenium browser automation | `04_selenium_image_scraper.ipynb` | Required because results are JS-rendered |
| Google Images | `google_images_download` + Selenium | `05_google_images_scraper.ipynb` | Supplementary top-up collection |

**Why Selenium rather than HTTP scraping:** image search results are rendered client-side. A plain
`requests` + BeautifulSoup approach returns an empty shell. Selenium drives a real browser, waits
for render, then scrapes the DOM. This worked but was **slow and fragile** — each source needed its
own CSS selector strategy, and those selectors broke as the sites changed. A meaningful share of
project time went to repairing scrapers rather than writing new code.

## Collection volumes

Approximate counts by collection batch, as they exist in the working directory:

| Batch | Images |
|---|---|
| Initial meme set | ~139 |
| DuckDuckGo batch | ~317 |
| Bing batch | ~146 |
| Supplementary scrape | ~602 |
| **Combined, pre-dedup** | **~1,700** |
| After perceptual-hash deduplication | ~1,690 distinct |
| Reaching annotation | **1,667** |
| **Final labeled dataset** | **1,320** |

The drop from 1,667 → 1,320 comes from label reconciliation: each item was labeled independently by
two people, and items where the two labels disagreed were dropped rather than adjudicated. That
removed ~21% of the corpus — and, importantly, the removed items are the *ambiguous* ones, so the
final dataset is the easy subset by construction. This bounds every metric reported later.

## The duplication problem

The single biggest surprise in collection: **the same memes appear everywhere.** Viral content is
re-hosted across Reddit, 4chan, and every image search index, so multi-source collection returns
enormous overlap. Naive aggregation would have produced a corpus that was largely repeats.

Why this is not merely wasteful but *actively dangerous*: duplicates that land on both sides of a
train/test split leak the test set. A model can score well by memorizing rather than generalizing,
and the reported accuracy becomes meaningless.

Handled with perceptual-hash deduplication — documented in
[`02_preprocessing.md`](02_preprocessing.md).

## Text data (for the baseline)

The text-only baseline did **not** use meme OCR text. It used the **FRENK hate-speech benchmark,
LGBT subset** — a public, pre-labeled English corpus loaded via HuggingFace `datasets`.

**This was a methodological mistake we did not recognize at the time.** Using a different corpus
for the baseline means the text-vs-multimodal comparison is not a controlled ablation: the two
models differ in *both* modality and training data. The correct design would have trained the text
baseline on the meme OCR captions alone. The reported ~7-point multimodal gain is therefore
indicative, not isolated. This is the most important caveat on the project's headline claim.

## Splits

70 / 15 / 15, giving roughly **924 train / 198 validation / 198 test**.

Two problems with this, in hindsight:

1. **The test set is tiny.** 198 examples means a single misclassification moves accuracy by ~0.5
   points. Differences between the three CLIP encoders — 84.85% / 89.73% / 93.94% — span a range
   where the smaller gaps may not be statistically separable. We reported point estimates with no
   confidence intervals.
2. **A single fixed split.** With a dataset this small, k-fold cross-validation was the right
   choice and would have cost almost nothing on frozen embeddings. We didn't do it.

## Class distribution

**80.9% positive / 19.1% negative** (1,068 / 252 of the final 1,320).

This is **inverted relative to real moderation traffic**, where the overwhelming majority of content
is benign. The consequence: a majority-class baseline already scores ~80.9% on this dataset, so
headline accuracy substantially overstates practical utility. Minority-class precision and recall
are the meaningful numbers.

The imbalance arose naturally from the collection strategy — we searched for hateful content, so we
found it. Correcting it would have required deliberately collecting benign LGBTQ+ memes at volume,
which we did partially but not enough.

## What we would do differently

1. **Train the text baseline on meme captions** rather than a separate corpus, so the
   text-vs-multimodal comparison is a genuine ablation. This is the highest-value fix.
2. **k-fold cross-validation** instead of a single fixed split — nearly free on cached embeddings,
   and the only way to put confidence intervals around the encoder comparison.
3. **Keep contested items rather than dropping them** — as soft labels, or as a dedicated hard-case
   evaluation set. Deleting them made the metrics look better and the dataset less useful.
4. **Collect a benign control set deliberately** to approach realistic class balance, instead of
   inheriting the 80.9/19.1 skew from the search strategy.
5. **Log deduplication parameters** (pHash threshold, pairs merged) so the corpus is reproducible.
6. **Use a VLM to pre-label at scale.** Labeling capacity was the binding constraint on dataset
   size, and dataset size bounded every result. Not available to us in 2024; entirely practical now.

---

**Next:** [`02_preprocessing.md`](02_preprocessing.md) — OCR, deduplication, and text repair.
