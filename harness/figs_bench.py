# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.8"]
# ///
"""Figures for Test B (same label, different files) and Test C (template swap)."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "img"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, ORANGE = "#2a78d6", "#eb6834"
GROUP_NAMES = {"ALL": "All 300 tasks", "gsm8k": "GSM8K maths", "ifeval": "IFEval instructions",
               "bfcl": "BFCL function calls", "multiturn": "Multi-turn rules", "toolresult": "Tool results"}

plt.rcParams.update({"font.family": "Arial", "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK, "axes.edgecolor": GRID,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def load(name: str) -> list[dict]:
    return list(csv.DictReader(open(ROOT / "results" / name)))


def _clean(ax) -> None:
    ax.tick_params(axis="both", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig_test_b() -> None:
    rows = [r for r in load("bench_testb.csv")]
    groups = ["ALL", "gsm8k", "ifeval", "bfcl", "multiturn", "toolresult"]
    runs = sorted({r["run"] for r in rows}, key=lambda n: ("q8" not in n, n))
    label = {r: "Qwen Q8_0 (reference)" if "official-q8" in r
             else "Qwen (official)" if r.split("__")[1] == "official" else r.split("__")[1] for r in runs}
    fig, axes = plt.subplots(1, len(groups), figsize=(15, 5.2), sharey=True)
    for ax, g in zip(axes, groups):
        for i, run in enumerate(runs):
            r = next((x for x in rows if x["run"] == run and x["group"] == g), None)
            if not r:
                continue
            s, lo, hi = float(r["score"]), float(r["ci_low"]), float(r["ci_high"])
            col = INK2 if "q8" in run else BLUE
            ax.plot([lo, hi], [i, i], color=col, lw=2, solid_capstyle="round", alpha=0.45)
            ax.plot(s, i, "o", color=col, ms=8, mec=SURFACE, mew=1.5)
            if g == "ALL":
                ax.text(0.62, i, f"{s:.1%}", va="center", fontsize=9, color=INK2)
        ax.set_title(GROUP_NAMES[g], loc="left", fontsize=11, pad=6)
        ax.set_xlim(0.6, 1.04)
        ax.set_xticks([0.6, 0.8, 1.0])
        ax.set_xticklabels(["60%", "80%", "100%"])
        _clean(ax)
    axes[0].set_yticks(range(len(runs)))
    axes[0].set_yticklabels([label[r] for r in runs], fontsize=10)
    axes[0].invert_yaxis()
    fig.suptitle("Same label, different files: Qwen3-8B Q4_K_M from eight uploaders", x=0.01, ha="left",
                 fontsize=14, fontweight="bold")
    fig.text(0.01, 0.01, "Dots are scores; lines are 95% Wilson intervals (axis starts at 60%). Same llama.cpp "
             "build, greedy decoding, fixed seed, thinking off. RTX 5090.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(IMG / "fig3_testb_files.png", dpi=200)


def fig_test_c(pairs: list[tuple[str, str, str]]) -> None:
    """pairs: (label, embedded run, swapped run)."""
    rows = load("bench_testc.csv")
    items = []
    for lab, a, b in pairs:
        for g in ("bfcl", "toolresult", "multiturn"):
            ra = next((x for x in rows if x["run"] == a and x["group"] == g), None)
            rb = next((x for x in rows if x["run"] == b and x["group"] == g), None)
            if ra and rb:
                items.append((f"{lab}: {GROUP_NAMES[g]}", ra, rb))
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(items) + 1.6))
    for i, (lab, ra, rb) in enumerate(items):
        sa, sb = float(ra["score"]), float(rb["score"])
        ax.plot([sa, sb], [i, i], color=GRID, lw=3, zorder=1)
        ax.plot(sb, i, "o", color=BLUE, ms=8, mec=SURFACE, mew=1.2, zorder=3)
        ax.plot(sa, i, "o", mfc="none", mec=ORANGE, ms=14, mew=2.2, zorder=4)
        p = rb.get("mcnemar_p", "")
        pv = "" if not p else ("  p < 0.001" if float(p) < 0.001 else f"  p = {float(p):.2f}")
        ax.text(1.03, i, f"{(sb - sa) * 100:+.0f} pts{pv}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([x[0] for x in items], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(-0.03, 1.25)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    _clean(ax)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="none", mec=ORANGE, mew=2.2, ms=12,
                              label="Template inside the file"),
                       Line2D([], [], marker="o", ls="", color=BLUE, ms=8, label="Other template swapped in")],
              loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, frameon=False, fontsize=10)
    ax.set_title("Same file, different template", loc="left", fontsize=14, fontweight="bold", pad=34)
    fig.text(0.01, 0.035, "Same weights in each pair; only the chat template changes. p: how likely a gap this big "
             "would be if both templates did equally well (exact McNemar test on paired items).", fontsize=8.5, color=MUTED)
    fig.text(0.01, 0.008, "Swaps: Mistral and Qwen2.5 files carry stale templates and get the current official "
             "one; the gpt-oss-20b file carries the official template and gets Unsloth's.", fontsize=8.5,
             color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(IMG / "fig4_testc_templates.png", dpi=200)


if __name__ == "__main__":
    IMG.mkdir(exist_ok=True)
    if (ROOT / "results" / "bench_testb.csv").exists():
        fig_test_b()
    if (ROOT / "results" / "bench_testc.csv").exists():
        fig_test_c([("Mistral-7B", "mistral-7b__lmstudio__embedded", "mistral-7b__lmstudio__official-template"),
                    ("Qwen2.5-7B", "qwen2.5-7b__maziyarpanahi__embedded",
                     "qwen2.5-7b__maziyarpanahi__official-template"),
                    ("gpt-oss-20b", "gpt-oss-20b__ggml-org__embedded", "gpt-oss-20b__ggml-org__unsloth-template")])
    print("figures done")
