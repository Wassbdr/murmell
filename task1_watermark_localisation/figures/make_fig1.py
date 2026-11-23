"""fig1_timeline.png: nested CV and public score of each step of REPORT.md, section 4 (numbers copied from that table).
    python figures/make_fig1.py        (needs matplotlib)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEPS = [  # name, short label, nested CV, public
    ("C4r", "synthetic U-Nets\n+ regime routing", 0.6795, 0.6843),
    ("C8", "decoder-ranked boxes", 0.7004, 0.6940),
    ("C9", "fold-clean full models\nin small cells", 0.7233, 0.7185),
    ("N00", "decoder-initialised\nsegmenter, PixelSeal", 0.7487, 0.7453),
    ("N01", "full-res decoder init,\nreal fine-tuning", 0.7698, 0.7665),
    ("S03", "per-image\nmask selector", 0.7828, 0.7823),
    ("S05", "crop-and-resize\naugmentation", 0.7884, 0.7845),
    ("S10", "pool pruning", 0.7959, 0.7953),
    ("S11", "fold-clean\nobject items", 0.8026, 0.7957),
    ("T01", "SAM contours", 0.8176, 0.8038),
    ("T02", "selector on top", 0.8205, 0.8050),
    ("B10", "second snapping\nroute", 0.8239, 0.8078),
    ("B11", "fourth chain", 0.8250, 0.8088),
    ("B21", "PixelSeal full:\nmean of 27 models", 0.8275, 0.8080),
    ("B22", "object-trained chain,\nsmall cells", 0.8275, 0.8123),
    ("B28", "second series of\nobject-trained models", 0.8285, 0.8127),
    ("B29", "full cells from\nthe selector", 0.8291, 0.8132),
]
FINAL = 0.8328  # official final ranking, hidden 70 %, Task 1

x = list(range(len(STEPS)))
cv = [s[2] for s in STEPS]
pub = [s[3] for s in STEPS]

fig, ax = plt.subplots(figsize=(13, 6.2))
ax.plot(x, cv, "o-", color="#1f77b4", label="nested CV, 510 train images (nested per stage, not end to end)")
ax.plot(x, pub, "s--", color="#ff7f0e", label="public leaderboard (30 % of test)")
ax.axhline(FINAL, color="#444444", lw=1, ls=":")
ax.text(-0.3, FINAL + 0.002, f"final score on the hidden 70 %: {FINAL:.4f} (best of our submissions; "
        "the ranking does not say which one)", fontsize=9, color="#444444", va="bottom")
for i, s in enumerate(STEPS):
    ax.text(i, max(cv[i], pub[i]) + 0.004, s[0], ha="center", va="bottom", fontsize=8, fontweight="bold" if s[0] == "B28" else "normal")
ref = [s[0] for s in STEPS].index("B28")
ax.plot([ref], [cv[ref]], "o", ms=13, mfc="none", mec="#1f77b4")
ax.text(ref, 0.790, "B28, reference:\nnested 0.8285\npublic 0.8127", ha="center", va="top", fontsize=8.5)
ax.set_xticks(x)
ax.set_xticklabels([s[1].replace("\n", " ") for s in STEPS], rotation=40, ha="right", fontsize=8.5,
                   rotation_mode="anchor")
ax.set_ylabel("mean per-image mIoU")
ax.set_ylim(0.67, 0.845)
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=9)
ax.set_title("Task 1, each step of REPORT.md section 4 (cumulative)", fontsize=11, loc="left")
fig.tight_layout()
fig.savefig(Path(__file__).with_name("fig1_timeline.png"), dpi=130)
