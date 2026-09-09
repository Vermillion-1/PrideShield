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

## Scraping: every method tried

Collection went through roughly six approaches before settling. Recording the failures because they
explain the final architecture better than the successes do.

### ❌ Attempt 1 — Twitter/X via `twint`

`twint` was the standard free Twitter scraper before the 2023 API changes. We cloned it and tried to
use it. **It no longer worked** — Twitter's endpoint changes had broken it, and the project was
effectively unmaintained. The official X API was paid-only at any useful volume.

**Outcome: X dropped entirely as a source.** This is a real gap — X is arguably where the most
meme-format hate speech circulates — and it was closed to us purely by cost.

### ❌ Attempt 2 — Commercial scraping API (ScraperAPI)

Prototyped a Reddit comment scraper against **ScraperAPI**, which handles proxy rotation and
CAPTCHA solving. The surviving script still has `scraper_api_key = 'YOUR API KEY'` in it — **it was
never funded**, so this path stopped at the prototype.

This is the clearest artifact of the budget constraint: the code exists, and simply could not be
run.

### ❌ Attempt 3 — Plain `requests` + BeautifulSoup on image search

The obvious cheap approach. **Returned nothing usable.** Image search results on Google, Bing and
DuckDuckGo are rendered client-side by JavaScript, so an HTTP fetch returns a shell page with no
image URLs in the markup.

This failure is what forced browser automation.

### ✅ Attempt 4 — Reddit via PRAW (free tier)

`01_reddit_scraper_praw.ipynb` — the official Python Reddit API Wrapper, keyword-targeted across
selected subreddits. Free tier, so **rate-limited**; collection ran across multiple sessions rather
than in one pass.

`02_reddit_scraper_async_praw.ipynb` — an `asyncpraw` rewrite to raise throughput within the same
rate limits by overlapping requests instead of blocking on each one.

**Worked well.** Reddit was the most reliable structured source.

### ✅ Attempt 5 — 4chan via public JSON API

`03_4chan_scraper.ipynb` — 4chan exposes threads, posts and attached media as **public JSON with no
authentication**. Straightforward `requests` + parse; no browser automation needed.

Highest hostile-content density of any source, and the easiest to collect from — the two facts are
related, and worth being uncomfortable about.

### ✅ Attempt 6 — Browser automation for image search

`04_selenium_image_scraper.ipynb` — **Selenium** driving a real Chrome instance: load the search
page, wait for JS render, scroll to trigger lazy loading, then scrape image URLs from the live DOM.

Run against **DuckDuckGo** and **Bing**, each needing its own CSS selector strategy. Also used
`bing_image_downloader` as a simpler wrapper for bulk Bing pulls, and `google_images_download` plus
Selenium for Google (`05_google_images_scraper.ipynb`).

**This worked, but it was the most fragile part of the pipeline:**
- Selectors broke whenever a provider changed its markup — repeatedly, over months.
- Lazy loading meant scroll timing mattered; scroll too fast and you collect nothing.
- Headless mode was detected and blocked by some providers, so runs needed a visible browser.
- Slow: a real browser rendering real pages, versus milliseconds for an API call.
- Required a matching `chromedriver` build, pinned per Chrome version.

**A meaningful share of project time went to repairing scrapers rather than writing new code.**

### Summary

| Method | Status | Why |
|---|---|---|
| Twitter/X via `twint` | ❌ Abandoned | Broken by API changes; official API paid-only |
| ScraperAPI (commercial) | ❌ Abandoned | No budget — never funded past prototype |
| `requests` + BeautifulSoup on image search | ❌ Failed | Results are JS-rendered; returns empty shell |
| Reddit PRAW / asyncpraw | ✅ Used | Free tier, rate-limited, reliable |
| 4chan public JSON API | ✅ Used | No auth, simple, high hostile-content density |
| Selenium + DDG/Bing/Google Images | ✅ Used | Only way to reach JS-rendered image results |

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
