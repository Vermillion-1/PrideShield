# 05 — Future Work

What the current state of the art offers that did not exist when this was built (July 2024 – May
2025), and how the project would be rebuilt today.

---

## Why the design is dated

Frozen CLIP embeddings + concatenation + a small head was a defensible 2024 design under a
zero-dollar budget. It is not what anyone should build now — and the parts that moved fastest are
precisely the parts this project struggled with: **multimodal fusion, OCR on stylized text, and
labeling throughput.**

## Where the state of the art is

### Instruction-tuned VLMs replaced embed-then-classify

Treating CLIP as a frozen feature extractor with a learned head was standard practice in 2022–2024.
It has largely been superseded by vision-language models that take an image and a prompt and reason
over both jointly:

| Family | Relevance |
|---|---|
| **Qwen2.5-VL / Qwen3-VL** | Strong open-weight VLMs, notably good OCR and text-in-image handling |
| **InternVL** | Competitive open-weight multimodal reasoning, strong on fine visual detail |
| **LLaVA / LLaVA-NeXT** | Reference open architecture for visual instruction tuning |
| **Molmo, Pixtral, Gemma-3 vision** | Recent open-weight entrants with permissive licensing |
| **Frontier APIs** (Claude, GPT, Gemini) | Strongest zero-shot multimodal reasoning, no training required |

The relevant capability is not better image classification — it is that these models **read the text
inside the image and reason about its relationship to the imagery in one pass.** That is exactly the
operation concatenated CLIP embeddings cannot perform.

### Benchmarks now exist for this task

At project start, hateful-meme detection had essentially one public benchmark (Meta's **Hateful
Memes Challenge**, 2020). The space is better served now: **HarMeme/Harm-P**, **MAMI** (misogynous
memes), **HatReD** (hateful meme *reasoning and explanation*), and multilingual efforts including
**HOMO-MEX** for Spanish-language LGBTQ+ hate.

A restart should evaluate on established public benchmarks first, treating a custom corpus as a
domain-specific supplement — the reverse of what was done here.

### PEFT removed the compute wall

The hardest constraint was compute: a 304M-parameter encoder could not be fine-tuned on free-tier
Colab, which is why encoders stayed frozen. **LoRA / QLoRA** now make it feasible to fine-tune
multi-billion-parameter VLMs on a single consumer GPU via low-rank adapters over a quantized base.
The constraint that shaped this entire architecture is largely gone.

## The rebuild, in order

**1. Establish the zero-shot baseline first.** Before training anything, measure a modern VLM
zero-shot and few-shot on the existing 1,320 examples, reporting macro metrics and AUPRC. This may
end the project — if zero-shot matches 93.94%, the trained pipeline has no justification. If it
doesn't, it becomes the bar every subsequent model must beat, which is far more meaningful than the
83.33% test-split floor. *This inverts the original workflow: 2024 was collect → train → evaluate; 2026 is
evaluate what exists → collect only where it fails.*

**2. Break the labeling bottleneck.** Dataset size bounded nearly every result, and labeling
throughput bounded dataset size. VLM pre-labeling with human adjudication of low-confidence cases
could plausibly cover an order of magnitude more data for the same effort. While doing it: keep
contested items as soft labels or a hard-case benchmark; deliberately collect benign content to
approach realistic class balance; version the dataset.

**3. Replace the fusion mechanism.** Fine-tune a VLM directly (LoRA/QLoRA) so the model attends
jointly over image and caption natively — the fusion problem dissolves rather than being engineered
around. If staying with frozen encoders for cost reasons, use cross-attention rather than
concatenation. Either way, **drop the OCR correction stack** and let the VLM read the image text;
modern VLMs handle stylized multi-panel meme text far better than 2024-era OCR, which removes one of
the largest engineering components of the original pipeline.

**4. Fix the evaluation protocol.** A like-for-like text-only ablation on the *same* corpus; k-fold
CV with confidence intervals; macro metrics and AUPRC as headline figures; multi-seed runs; a test
set at realistic class balance; confusion-matrix and qualitative error analysis.

**5. Evaluate what gates deployment.** False-positive rate on reclaimed in-group language — the
dominant failure mode, and the one where errors do the most harm. Adversarial robustness (character
substitution, leetspeak, text-in-image obfuscation, crops, re-encoding). Subgroup breakdowns.
Calibration, so uncertain cases can be routed to humans.

**6. Explanation, not just classification.** A VLM can be prompted to *explain why* a meme is
hateful, producing an auditable rationale. For moderation — where decisions are appealed and must be
justified — this is far more useful than a scalar, and it addresses the interpretability gap
directly. The **HatReD** dataset targets exactly this.

**7. Infrastructure from day one.** Experiment tracking (W&B/MLflow), config-driven sweeps, seeded
and versioned runs tied to dataset snapshots.

## What carries forward

Not everything is superseded. These hold regardless of tooling:

- **Deduplicate before splitting.** Near-duplicate leakage invalidates results under any
  architecture; perceptual hashing is still the right tool.
- **Benchmark encoders under identical downstream conditions.** The `RN50x64` finding only holds
  because the comparison was controlled.
- **Treat the loss function as a search parameter** rather than assuming Focal Loss suits imbalance —
  it was selected for one encoder and not another.
- **Independent dual labeling** on subjective tasks, treating the disagreement rate as a measurement
  of task difficulty rather than an inconvenience.
- **Quote accuracy against the majority-class floor of the split you actually evaluated** — not the
  corpus-level figure, which can differ when splits are unstratified. A discipline that survives any
  architecture change.
- **Stratify your splits.** It costs nothing and removes an entire class of misreading.

---

**Previous:** [`04_evaluation.md`](04_evaluation.md) · **Back to:** [`../README.md`](../README.md)
