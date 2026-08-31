#!/usr/bin/env python3
"""Fig. 1 def — DIC / Ca2+ / CRAM-like % regressions.

Global-sheet DIC and Ca are µmol L-1; converted to mg L-1 here.

python fig1/plot_def.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "global_pearl_puding data.xlsx"
OUT = ROOT / "figures" / "fig1_def.png"

COLOR_PEARL = "#8B1E3F"
COLOR_GLOBAL = "#F2B8B0"
COLOR_ALL_LINE = "#333333"


def format_p(p: float) -> str:
    if p < 0.01:
        return "p < 0.01"
    if p < 0.05:
        return "p < 0.05"
    return f"p = {p:.2f}"


def load():
    pearl = pd.read_excel(DATA, "pearl river").rename(
        columns={"CRAM(%)": "CRAM", "DIC(mg/L)": "DIC", "Ca2+(mg/L)": "Ca"}
    )
    glob = pd.read_excel(DATA, "global").rename(
        columns={"CRAM(%)": "CRAM", "DIC(mg/L)": "DIC", "Ca2+(mg/L)": "Ca"}
    )
    for df in (pearl, glob):
        for c in ("CRAM", "DIC", "Ca"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    glob["DIC"] = glob["DIC"] * 12.011 / 1000.0
    glob["Ca"] = glob["Ca"] * 40.078 / 1000.0
    return pearl, glob


def fit_ci(x, y, n=200):
    slope, intercept, _, _, _ = linregress(x, y)
    xl = np.linspace(x.min(), x.max(), n)
    yl = slope * xl + intercept
    xm, sxx = x.mean(), np.sum((x - x.mean()) ** 2)
    sxy = np.sum((x - x.mean()) * (y - y.mean()))
    syy = np.sum((y - y.mean()) ** 2)
    syx = np.sqrt(max((syy - sxy**2 / sxx) / (len(x) - 2), 0))
    t = stats.t.ppf(0.975, len(x) - 2)
    se = syx * np.sqrt(1 / len(x) + (xl - xm) ** 2 / sxx)
    return xl, yl, yl - t * se, yl + t * se


def panel(ax, pearl, glob, x, y, xlabel, ylabel, pearl_only, xlim, ylim, letter):
    if not pearl_only:
        g = glob[[x, y]].dropna()
        ax.scatter(g[x], g[y], s=18, c=COLOR_GLOBAL, alpha=0.7, edgecolors="none", zorder=2)
        xl, yl, lo, hi = fit_ci(g[x].to_numpy(), g[y].to_numpy())
        ax.fill_between(xl, lo, hi, color="#C8C8C8", alpha=0.4, zorder=1)
        ax.plot(xl, yl, color=COLOR_ALL_LINE, ls="--", lw=1.6, zorder=3)
        r, p = stats.pearsonr(g[x], g[y])
        ax.text(0.04, 0.86, f"All Samples: r = {r:.2f}, {format_p(p)}",
                transform=ax.transAxes, va="top", fontsize=9, fontweight="bold", color=COLOR_ALL_LINE)

    p = pearl[[x, y]].dropna()
    ax.scatter(p[x], p[y], s=20, c=COLOR_PEARL, alpha=0.85, edgecolors="none", zorder=4)
    xl, yl, lo, hi = fit_ci(p[x].to_numpy(), p[y].to_numpy())
    ax.fill_between(xl, lo, hi, color="#E6C4CB", alpha=0.45, zorder=3)
    ax.plot(xl, yl, color=COLOR_PEARL, lw=1.8, zorder=5)
    r, pv = stats.spearmanr(p[x], p[y])
    ax.text(0.04, 0.96, f"Pearl River: r = {r:.2f}, {format_p(pv)}",
            transform=ax.transAxes, va="top", fontsize=9, fontweight="bold", color=COLOR_PEARL)

    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=9)
    ax.text(-0.12, 1.04, f"{letter})", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom")


def main() -> None:
    pearl, glob = load()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Myriad Pro", "Arial"],
        "axes.linewidth": 1.1,
        "axes.grid": False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0), dpi=200)
    panel(axes[0], pearl, glob, "DIC", "Ca", "DIC (mg/L)", r"Ca$^{2+}$ (mg/L)",
          True, (0, 65), (0, 115), "d")
    panel(axes[1], pearl, glob, "DIC", "CRAM", "DIC (mg/L)", "CRAM-like %",
          False, (0, 75), (40, 60), "e")
    panel(axes[2], pearl, glob, "Ca", "CRAM", r"Ca$^{2+}$ (mg/L)", "CRAM-like %",
          False, (0, 100), (40, 60), "f")
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_PEARL,
                   markersize=8, label="Pearl River Data"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_GLOBAL,
                   markersize=8, label="Global Data"),
        ],
        loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.08),
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", OUT)


if __name__ == "__main__":
    main()
