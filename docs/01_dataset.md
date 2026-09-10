# 01 — Dataset Construction

Building a labeled corpus of anti-LGBTQ+ memes from scratch, with no budget.

---

## Why build a dataset

No public corpus fit the task. The adjacent resources each failed in a specific way:

- **Hateful Memes Challenge (Meta)** — synthetically constructed, covers hate broadly rather than
  the LGBTQ+-directed subset.
- **Text-only hate corpora** (Davidson, FRENK, HateXplain) — cannot support a multimodal task.
  FRENK's LGBT subset was used for the text baseline only.
- **Platform moderation datasets** — not accessible without institutional partnerships.

## The constraint: no funding

No paid API tier, no annotation vendor, no compute grant. This determined the architecture more than
any modeling decision:

| Wanted | Blocked by | Done instead |
|---|---|---|
| Twitter/X collection | Paid-only after 2023 API changes | X dropped as a source |
| Bulk image-search APIs | No free tier at useful volume | Selenium browser automation |
| Commercial annotation | Cost per label | Hand-labeled by two project members |
| GPU compute | No grant | Colab free tier → frozen encoders, small head |

The constraint wasn't purely limiting: frozen embeddings made the pipeline fast and reproducible.
The real cost was to *dataset scale*, which bounds every result.

## Collection: six methods, three failed

### ❌ Twitter/X via `twint`
`twint` was the standard free Twitter scraper before the 2023 API changes. Cloned it; **it no longer
worked** — endpoint changes had broken it and the project was unmaintained. The official API was
paid-only at useful volume. **X dropped entirely** — a real gap, since it's arguably where the most
meme-format hate circulates.

### ❌ Commercial scraping API (ScraperAPI)
Prototyped a Reddit scraper against ScraperAPI for proxy rotation and CAPTCHA handling. The script
still contains `scraper_api_key = 'YOUR API KEY'` — **never funded**, so it stopped at prototype.
The clearest artifact of the budget constraint: the code exists and could not be run.

### ❌ `requests` + BeautifulSoup on image search
The cheap approach. **Returned nothing usable** — image search results are rendered client-side, so
an HTTP fetch returns a shell page with no image URLs in the markup. *This failure is what forced
browser automation.*

### ✅ Reddit — PRAW / asyncpraw
Official API wrapper, keyword-targeted across selected subreddits. Free tier, so rate-limited;
collection ran across multiple sessions. An `asyncpraw` rewrite overlapped requests to raise
throughput within the same quota. Most reliable structured source.

### ✅ 4chan — public JSON API
Threads, posts and media exposed as public JSON with no authentication. Straightforward `requests` +
parse. Highest hostile-content density of any source and the easiest to collect from — the two facts
are related.

### ✅ Image search — Selenium
Drives a real Chrome instance: load search page, wait for JS render, scroll to trigger lazy loading,
scrape image URLs from the live DOM. Run against DuckDuckGo and Bing (plus `bing_image_downloader`
for bulk pulls, and `google_images_download` for Google).

Worked, but was the most fragile component: per-provider CSS selectors that broke on markup changes,
scroll timing dependencies, headless detection forcing a visible browser, and a `chromedriver` build
pinned per Chrome version. **A meaningful share of project time went to repairing scrapers.**

## Volumes

| Stage | Count |
|---|---|
| After perceptual-hash deduplication | **1,696** (476 MB) |
| Reaching OCR extraction | 1,405 |
| Reaching labeling | 1,667 |
| **Final dataset** | **1,320** |

The pre-deduplication raw count was not preserved in the surviving artifacts, so no deduplication
rate is claimed. OCR coverage of 1,405 of 1,696 images is also unexplained by what survives — most
likely unreadable or non-meme files removed before extraction — and is recorded rather than smoothed
over. The 1,667 → 1,320 drop is broken down under
[Splits and class balance](#splits-and-class-balance) below.

## Deduplication

The same viral memes appear on every platform, so multi-source collection returns heavy overlap.
This is not merely wasteful — **duplicates spanning a train/test boundary leak the test set**, letting
a model score well by memorizing. Handled with perceptual hashing; see
[`02_preprocessing.md`](02_preprocessing.md).

## Splits and class balance

**70/15/15** → 924 train / 198 validation / 198 test.

**Class distribution: 80.9% positive / 19.1% negative** (1,068 / 252). This is inverted relative to
real moderation traffic, where most content is benign, so a majority-class baseline already scores
well and accuracy must be read against that floor. The skew arose naturally from the collection
strategy: we searched for hateful content, so we found it.

**The splits are not stratified**, and the balance drifts across them:

| Split | Rows | Hateful | Not hateful | % hateful |
|---|---|---|---|---|
| Train | 924 | 737 | 187 | 79.76% |
| Validation | 198 | 166 | 32 | 83.84% |
| **Test** | **198** | **165** | **33** | **83.33%** |

So the reference floor for test accuracy is **83.33%**, not the corpus-level 80.9%. Stratified
splitting would have cost nothing and should have been used.

### How 1,667 became 1,320

| Stage | Count | Share |
|---|---|---|
| Dual-annotated | 1,667 | 100% |
| Annotators agreed | 1,448 | **86.9%** |
| Dropped — annotators disagreed | 219 | 13.1% |
| Dropped — both marked "cannot judge" (`-1`) | 110 | 6.6% |
| Dropped — no usable image or caption downstream | 18 | 1.1% |
| **Final dataset** | **1,320** | **79.2%** |

Annotation used three codes — hateful (`1`), not hateful (`0`), and a third code for items that
could not be judged (`-1`). Dropping the 219 disagreements rather than adjudicating them removes
precisely the ambiguous cases, so the final corpus skews toward clear-cut examples and every
accuracy figure is an accuracy on an easier distribution than the real one.

## Text data (baseline only)

The text-only baseline used the public **FRENK hate-speech benchmark, LGBT subset** via HuggingFace
`datasets` — *not* meme OCR captions.

---

## Notes & Caveats

<sub>

**The baseline corpus mismatch is the most consequential issue here.** Training the text baseline on
FRENK rather than on meme captions means the text-vs-multimodal comparison varies both modality and
training data, so the reported multimodal gain is indicative rather than a controlled ablation.
Fixing it would have been cheap.

**Splits.** A ~198-example test set means one misclassification moves accuracy ~0.5 points, and
encoder gaps of a few points may not be statistically separable. A single fixed split was used where
k-fold cross-validation was the right protocol — and would have cost minutes on cached embeddings.

**Labels.** Dropping contested items removed exactly the ambiguous cases, making the final dataset
the easy subset by construction and inflating all downstream metrics. Soft labels or a dedicated
hard-case evaluation set would have been better. Both annotators were project members; no Cohen's κ
was computed despite both label columns being retained.

**Reproducibility.** Deduplication parameters (pHash threshold, pairs merged) were not logged, so
the corpus is not exactly reproducible from the notebooks alone.

**Coverage.** English-only, narrow platform and time window, and no deliberate collection of benign
LGBTQ+ content to counter the class skew.

</sub>

---

**Next:** [`02_preprocessing.md`](02_preprocessing.md)
