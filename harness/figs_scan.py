# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.8"]
# ///
"""Figures for Test A: what each engine builds, and the Qwen3-8B stale template timeline."""

import csv
import json
import math
import random
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import dates as mdates

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "img"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
SAME, DIFF, ERR = "#d6d5cf", "#ec835a", "#d03b3b"
FAMILIES = [("mistralai/Mistral-7B-Instruct-v0.3", "Mistral-7B-Instruct v0.3"),
            ("Qwen/Qwen3-8B", "Qwen3-8B"), ("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct"),
            ("openai/gpt-oss-20b", "gpt-oss-20b"), ("microsoft/Phi-4-mini-instruct", "Phi-4-mini-instruct"),
            ("Qwen/Qwen3.8-27B", "Qwen3.8-27B")]

plt.rcParams.update({"font.family": "Arial", "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK, "axes.edgecolor": GRID,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def fig_engines() -> None:
    s = json.loads((ROOT / "results" / "engine_summary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    titles = {"python": "Python Jinja (transformers, llama-cpp-python)", "llama": "llama.cpp (llama-server)"}
    for ax, eng in zip(axes, ("python", "llama")):
        for i, (fam, label) in enumerate(FAMILIES):
            d = s[fam][eng]
            parts = [("Error", d.get("crash", 0), ERR), ("Different prompt", d.get("different", 0), DIFF),
                     ("Same as official", d.get("same", 0), SAME)]
            left = 0.0
            for name, w, col in parts:
                if w <= 0:
                    continue
                ax.barh(i, w, left=left, height=0.62, color=col, edgecolor=SURFACE, linewidth=2)
                if w >= 0.09:
                    ax.text(left + w / 2, i, f"{w:.0%}", ha="center", va="center", fontsize=10,
                            color="#ffffff" if col == ERR else INK)
                left += w
        ax.set_xlim(0, 1)
        ax.set_title(titles[eng], loc="left", fontsize=11.5, color=INK, pad=8)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.tick_params(axis="both", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    axes[0].invert_yaxis()
    axes[0].set_yticks(range(len(FAMILIES)))
    axes[0].set_yticklabels([f"{lab}\n{s[f]['downloads_30d']:,} downloads" for f, lab in FAMILIES],
                            fontsize=10)
    from matplotlib.patches import Patch
    fig.legend([Patch(color=c) for c in (SAME, DIFF, ERR)], ["Same as official", "Different prompt", "Error or tools dropped"],
               loc="upper left", ncol=3, frameon=False, bbox_to_anchor=(0.005, 0.93), fontsize=10)
    fig.suptitle("Share of GGUF downloads that build the official prompt, by engine", x=0.01, ha="left",
                 fontsize=14, fontweight="bold", y=0.99)
    fig.text(0.01, 0.045, "Pure repacks only, weighted by Hub downloads in the last 30 days (23 Nov 2026). "
             "12 test chats per template. Uploads with no template (under 1%) not shown.",
             fontsize=8.5, color=MUTED)
    fig.text(0.01, 0.01, "Error in llama.cpp (Mistral): tool definitions are dropped without a warning, "
             "and chats that contain a tool result are rejected with HTTP 400.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.07, 1, 0.88))
    fig.savefig(IMG / "fig1_engines.png", dpi=200)


def fig_timeline() -> None:
    rows = [r for r in csv.DictReader(open(ROOT / "results" / "scan.csv"))
            if r["family"] == "Qwen/Qwen3-8B" and r["pure_repack"] == "True" and r["tpl_hash"]]
    rng = random.Random(7)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ys = {"stale": 1, "same": 0}
    for r in rows:
        cls = "stale" if r["cls"].startswith("stale") else "same" if r["cls"] == "same" else None
        if cls is None:
            continue
        x = date.fromisoformat(r["created"])
        y = ys[cls] + rng.uniform(-0.32, 0.32)
        size = 14 + 9 * math.sqrt(max(1, int(r["downloads"]))) / 10
        ax.scatter(x, y, s=size, color=ERR if cls == "stale" else "#9b9a94", alpha=0.75,
                   edgecolors=SURFACE, linewidths=1.2, zorder=3)
        if r["repo"] == "Qwen/Qwen3-8B-GGUF":
            ax.annotate("Qwen's own GGUF\n848,856 downloads", (x, y), xytext=(28, -38),
                        textcoords="offset points", fontsize=9.5, color=INK2,
                        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    fix = date(2025, 5, 19)
    ax.axvline(fix, color=INK2, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.text(fix, 1.52, "  19 May 2025: Qwen fixes the template in the base repo", fontsize=9.5,
            color=INK2, va="center")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Current template", "Stale template\n(Python Jinja fails on\nnull-content tool calls)"])
    ax.set_ylim(-0.6, 1.8)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.tick_params(axis="both", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    late = [r for r in rows if r["cls"].startswith("stale") and date.fromisoformat(r["created"]) > fix]
    ax.text(date(2026, 3, 1), 1.52, f"{len(late)} uploads made after the fix still carry the old template",
            fontsize=9.5, color=ERR, va="center", ha="left")
    ax.set_title("Qwen3-8B GGUF uploads by creation date: the old template still appears in new uploads",
                 loc="left", fontsize=14, fontweight="bold", pad=12)
    fig.text(0.01, 0.01, "Each dot is one pure repack; area grows with 30-day downloads. "
             "Stale = template from 28 Apr 2025.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(IMG / "fig2_qwen3_timeline.png", dpi=200)


if __name__ == "__main__":
    IMG.mkdir(exist_ok=True)
    fig_engines()
    fig_timeline()
    print("saved", sorted(p.name for p in IMG.glob("*.png")))
