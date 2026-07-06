"""Figures for technical_report_v2 — plotted from the report material's exact numbers."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path("docs/report/figures/udd_ablation")
OUT.mkdir(parents=True, exist_ok=True)
INK, MUT = "#1a1a1a", "#666666"
plt.rcParams.update({"font.size": 8, "text.color": INK, "axes.edgecolor": "#bbbbbb",
                     "xtick.color": INK, "ytick.color": INK})

ARMS = ["A1_spotting_off", "A1_spotting_on", "A2_reason_answer", "A2_reason_chain",
        "A3_base", "A3_reason", "A3_spot", "A3_spot_reason",
        "A4_en", "A4_en_ar", "A4_en_de", "A4_en_fr", "A4_en_id", "A4_en_ja", "A4_en_ko", "A4_en_zh",
        "A5_connector", "A5_llm_attn", "A5_llm_mlp", "A5_vision",
        "A6_r16", "A6_r32", "A6_r64", "A6_r8"]


def heatmap(path, data, rows, cols, title, vlim, w=6.2, cell_fs=6.8):
    data = np.array(data)
    fig, ax = plt.subplots(figsize=(w, 0.26 * len(rows) + 1.3))
    im = ax.imshow(data, cmap="RdBu", vmin=-vlim, vmax=vlim, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=7.5)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=7, family="monospace")
    for i in range(len(rows)):
        for j in range(len(cols)):
            dark = abs(data[i, j]) > 0.62 * vlim
            ax.text(j, i, f"{data[i, j]:+.3f}" if vlim < 0.5 else f"{data[i, j]:+.2f}",
                    ha="center", va="center", fontsize=cell_fs,
                    color="white" if dark else INK, family="monospace")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=9, loc="left", pad=8)
    cb = fig.colorbar(im, fraction=0.025, pad=0.02)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# ---- v1 coarse per-suite delta (Section 4)
V1 = [[+0.000, +0.065, +0.048, +0.123], [-0.167, -0.018, +0.073, +0.122],
      [+0.068, -0.028, -0.006, -0.077], [-0.167, -0.028, +0.014, -0.045],
      [+0.000, +0.065, +0.036, +0.119], [+0.000, -0.023, +0.058, +0.095],
      [-0.167, -0.012, +0.073, +0.099], [+0.000, -0.027, +0.068, +0.135],
      [+0.000, +0.045, +0.046, +0.104], [+0.000, +0.042, +0.073, +0.083],
      [+0.000, -0.025, +0.047, +0.060], [+0.000, -0.026, +0.060, +0.080],
      [-0.167, -0.090, +0.016, +0.054], [+0.000, +0.041, +0.059, +0.131],
      [-0.167, -0.159, +0.026, +0.089], [+0.000, -0.069, +0.047, +0.045],
      [-0.167, -0.088, +0.014, -0.030], [+0.000, -0.026, +0.054, +0.102],
      [+0.000, +0.041, +0.060, +0.099], [-0.167, -0.159, +0.025, +0.041],
      [+0.000, -0.027, +0.059, +0.136], [-0.167, -0.015, +0.069, +0.117],
      [-0.167, +0.039, +0.032, +0.082], [+0.000, -0.025, +0.085, +0.105]]
heatmap(OUT / "udd_full_coarse_delta.png", V1, ARMS,
        ["capability", "spatial", "realistic", "heldout"],
        "v1 — per-suite score delta (after − before), 24 arms", 0.17, w=5.4)

# ---- v2 task transfer (Section 5a)
V2T = [[+0.46, +0.25, +0.08, -0.01, +0.00, +0.00, +0.10], [+0.25, +0.21, -0.00, -0.01, +0.00, +0.04, +0.06],
       [+0.00, -0.08, -0.00, +0.00, +0.00, +0.00, -0.03], [+0.00, +0.00, -0.00, -0.00, +0.00, +0.00, -0.00],
       [+0.50, +0.25, +0.09, +0.00, +0.00, +0.00, +0.10], [+0.33, +0.25, +0.02, -0.00, +0.00, +0.00, +0.07],
       [+0.21, +0.17, +0.00, -0.01, +0.00, +0.04, +0.05], [+0.29, +0.17, +0.04, -0.01, +0.00, +0.04, +0.06],
       [+0.00, +0.25, -0.00, +0.00, +0.00, +0.12, +0.04], [+0.00, +0.17, -0.00, +0.00, +0.00, +0.00, +0.01],
       [+0.00, +0.21, -0.00, +0.00, +0.00, +0.00, +0.02], [+0.00, +0.21, -0.00, +0.00, +0.00, +0.04, +0.03],
       [+0.00, +0.12, +0.03, +0.00, +0.00, +0.04, +0.01], [+0.00, +0.21, -0.00, +0.00, +0.00, +0.04, +0.03],
       [+0.00, +0.17, -0.00, -0.00, +0.00, +0.00, +0.02], [+0.00, +0.12, -0.00, +0.00, +0.00, +0.00, +0.01],
       [+0.00, +0.04, -0.00, -0.00, +0.00, +0.00, -0.00], [+0.00, +0.12, +0.03, -0.01, +0.00, +0.04, +0.01],
       [+0.21, +0.17, +0.04, -0.00, +0.00, +0.04, +0.04], [+0.00, +0.12, -0.00, +0.00, +0.00, +0.00, +0.01],
       [+0.33, +0.17, +0.02, -0.01, +0.00, +0.04, +0.06], [+0.54, +0.25, -0.00, -0.01, +0.00, +0.00, +0.09],
       [+0.33, +0.17, +0.07, -0.00, +0.00, +0.04, +0.06], [+0.00, +0.25, +0.03, -0.01, +0.00, +0.08, +0.03]]
heatmap(OUT / "udd_v2_task_transfer.png", V2T, ARMS,
        ["vqa", "reason", "kie", "recogn", "table", "classi", "ALL"],
        "v2 — UDD heldout delta per task (~24 samples/task)", 0.55, w=6.0)

# ---- v2 region transfer (Section 5b)
R_ARMS = ["A1_spotting_off", "A1_spotting_on", "A3_base", "A3_reason", "A3_spot",
          "A3_spot_reason", "A5_connector", "A5_llm_attn", "A5_llm_mlp", "A5_vision"]
V2R = [[-0.02, -0.01, -0.02, +0.02], [+0.00, +0.00, -0.00, +0.00],
       [-0.02, +0.01, -0.02, +0.02], [-0.04, +0.01, -0.01, -0.00],
       [+0.00, +0.01, +0.00, +0.01], [-0.01, -0.02, -0.01, -0.02],
       [-0.07, -0.01, -0.02, +0.02], [-0.05, -0.06, -0.02, -0.03],
       [-0.10, -0.05, -0.04, -0.03], [-0.08, -0.04, -0.04, -0.01]]
heatmap(OUT / "udd_v2_region_transfer.png", V2R, R_ARMS,
        ["top-left", "top-right", "bottom-left", "bottom-right"],
        "v2 — localization grounding delta by document quadrant", 0.10, w=5.2)

# ---- data geometry scatter (Section 3): median dims, Okabe-Ito by task (fixed order)
SRC = [  # name, w, h, task
    ("ai2d", 521, 404, "vqa"), ("charxiv", 1000, 660, "reasoning"),
    ("doclaynet", 1000, 1000, "localization"), ("docvqa", 776, 1000, "vqa"),
    ("dvqa", 448, 448, "reasoning"), ("iam", 1000, 65, "recognition"),
    ("im2latex", 289, 56, "recognition"), ("infovqa", 417, 1000, "vqa"),
    ("latexocr", 240, 40, "recognition"), ("mathvista", 448, 369, "reasoning"),
    ("ocrbench", 184, 67, "vqa"), ("ocrvqa", 333, 500, "vqa"),
    ("plotqa", 1000, 604, "reasoning"), ("publaynet", 612, 792, "localization"),
    ("pubtabnet", 503, 161, "table"), ("sroie", 78, 19, "kie"),
    ("stvqa", 500, 333, "vqa"), ("synthdog_en", 897, 856, "recognition"),
    ("synthdog_ko", 905, 871, "recognition"), ("tatqa", 699, 226, "reasoning"),
    ("textvqa", 1000, 750, "vqa"), ("mtvqa", 1000, 750, "vqa"),
    ("ocrbench_v2", 562, 1000, "vqa"), ("omnidocbench", 708, 1000, "recognition"),
    ("rvl_cdip", 754, 1000, "classification"), ("chartqa", 850, 600, "reasoning"),
    ("screenqa", 562, 1000, "vqa"), ("docmatix", 773, 1000, "vqa"),
    ("hallusionbench", 696, 502, "reasoning"), ("visualmrc", 571, 1000, "vqa"),
    ("cord", 667, 1000, "kie"), ("funsd", 754, 1000, "kie")]
TASKS = ["vqa", "reasoning", "recognition", "localization", "kie", "table", "classification"]
OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]
fig, ax = plt.subplots(figsize=(6.0, 4.4))
for t, c in zip(TASKS, OKABE):
    pts = [(w, h, n) for n, w, h, tt in SRC if tt == t]
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=42, color=c,
               edgecolor="#333333", linewidth=0.5, label=t, zorder=3)
for n, w, h, _ in SRC:
    offsets = {"sroie": (4, 4), "latexocr": (2, -9), "im2latex": (4, 5), "iam": (-8, 6),
               "ocrbench": (-4, 7), "doclaynet": (-30, 7), "textvqa": (-16, 7),
               "synthdog_ko": (6, 2), "pubtabnet": (6, 2), "tatqa": (6, 2)}
    if n in offsets:
        ax.annotate(n, (w, h), textcoords="offset points", xytext=offsets[n],
                    fontsize=6.5, color=MUT)
ax.axhline(512, color="#999999", lw=0.8, ls="--", zorder=1)
ax.axvline(512, color="#999999", lw=0.8, ls="--", zorder=1)
ax.text(1005, 520, "512px training cap", fontsize=6.5, color=MUT, ha="right")
ax.set_xlabel("median width (px)", fontsize=8); ax.set_ylabel("median height (px)", fontsize=8)
ax.set_title("UDD source geometry — single-line crops vs full pages", fontsize=9, loc="left")
ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
ax.grid(color="#eeeeee", lw=0.6, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "udd_data_geometry.png", dpi=200); plt.close(fig)

# ---- QA density bar (Section 3): QAs per image, one sequential hue
QA = [("dvqa", 300, 1500), ("plotqa", 300, 1500), ("tatqa", 300, 1499), ("docmatix", 287, 1441),
      ("ocrvqa", 300, 1417), ("mtvqa", 299, 990), ("visualmrc", 278, 947),
      ("hallusionbench", 286, 574), ("stvqa", 300, 394), ("ocrbench_v2", 299, 301),
      ("chartqa", 296, 300), ("screenqa", 294, 298)]
rest = 20  # remaining sources are 1 QA/image
names = [n for n, _, _ in QA] + [f"(+{rest} sources)"]
ratio = [q / i for _, i, q in QA] + [1.0]
fig, ax = plt.subplots(figsize=(6.0, 3.2))
ax.barh(range(len(names)), ratio, height=0.62, color="#0072B2", zorder=3)
ax.invert_yaxis()
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
for i, r in enumerate(ratio[:5]):
    ax.text(r - 0.08, i, f"{r:.1f}", va="center", ha="right", fontsize=7, color="white",
            family="monospace", zorder=4)
ax.set_xlabel("QAs per image (folded into native lists)", fontsize=8)
ax.set_title("QA density — multi-QA sources vs the 1-QA majority", fontsize=9, loc="left")
ax.grid(axis="x", color="#eeeeee", lw=0.6, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "udd_data_qa_density.png", dpi=200); plt.close(fig)

print("wrote 5 figures ->", OUT)
