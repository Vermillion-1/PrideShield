"""Regenerate docs/figures/results.png.

Encoder parameter counts are the CLIP *vision tower* sizes, computed with
open_clip (`sum(p.numel() for p in model.visual.parameters())`), not estimated:

    ViT-B/32         87.8M   (full model 151.3M)
    ViT-L/14@336px  304.3M   (full model 427.9M)
    RN50x64         420.4M   (full model 623.3M)

The baseline drawn is the majority-class floor *on the test split actually
used* (165 positive / 33 negative = 83.33%), not the corpus-level 80.9%.

Usage: python docs/figures/make_results_figure.py
"""

from pathlib import Path

import matplotlib.pyplot as plt

TEST_FLOOR = 83.33  # 165 / 198 -- majority floor on the test split
OUT = Path(__file__).resolve().parent / "results.png"

BARS = [
    ("Majority\nbaseline", TEST_FLOOR, "#c9c4bd"),
    ("RN50x64\n(CNN)", 84.85, "#dd9080"),
    ("Text-only\n(BERT, FRENK)", 86.87, "#b0aab6"),
    ("ViT-B/32", 89.73, "#8fb3cc"),
    ("ViT-L/14\n@336px", 93.94, "#4a7d9e"),
]

# (label, vision-tower params in millions, accuracy, colour)
POINTS = [
    ("ViT-B/32", 87.8, 89.73, "#8fb3cc"),
    ("ViT-L/14@336px", 304.3, 93.94, "#4a7d9e"),
    ("RN50x64", 420.4, 84.85, "#dd9080"),
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#8e8c99",
    "axes.labelcolor": "#2b2a30",
    "text.color": "#2b2a30",
    "xtick.color": "#61606c",
    "ytick.color": "#61606c",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

# ---- left: model comparison ------------------------------------------------
labels = [b[0] for b in BARS]
values = [b[1] for b in BARS]
colors = [b[2] for b in BARS]

ax1.bar(labels, values, color=colors, width=0.62)
for x, v in enumerate(values):
    ax1.text(x, v + 0.35, f"{v:.2f}", ha="center",
             fontweight="bold" if v == 93.94 else "normal", fontsize=11)

ax1.axhline(TEST_FLOOR, ls="--", lw=1.2, color="#8e8c99", zorder=0)
ax1.text(-0.42, TEST_FLOOR - 1.1,
         f"majority-class floor on the test split — {TEST_FLOOR:.2f}%",
         fontsize=9.5, style="italic", color="#61606c")

ax1.set_ylim(78, 97.5)
ax1.set_ylabel("Accuracy (%)")
ax1.set_title("Model comparison", fontweight="bold", loc="left", fontsize=13)
ax1.spines[["top", "right"]].set_visible(False)
ax1.tick_params(labelsize=10)

# ---- right: params vs accuracy ---------------------------------------------
for name, params, acc, color in POINTS:
    ax2.scatter(params, acc, s=190, color=color, zorder=3)
    ax2.annotate(f"{name}\n{params:.0f}M",
                 (params, acc), textcoords="offset points",
                 xytext=(0, 16 if acc > 88 else -34),
                 ha="center", fontsize=10)

ax2.set_xlim(20, 480)
ax2.set_ylim(81.5, 96.5)
ax2.set_xlabel("Vision-encoder parameters (M)")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("More parameters ≠ better representations",
              fontweight="bold", loc="left", fontsize=13)
ax2.grid(alpha=0.25, color="#c9c4bd")
ax2.set_axisbelow(True)
ax2.spines[["top", "right"]].set_visible(False)
ax2.tick_params(labelsize=10)

fig.tight_layout()
fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
