#!/usr/bin/env python3
"""Fig. 3 — Puding CRAM vs DIC / Ca2+ / Δ14C.

python fig3/plot_fig3.py
"""

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "global_pearl_puding data.xlsx"
C14 = Path(__file__).resolve().parent / "fig3_data.xlsx"
OUT = ROOT / "figures" / "fig3.png"
FONT_DIR = Path("/home/pangyu/myriad-pro")
FONT_BOLD = FONT_DIR / "MYRIADPRO-BOLD.OTF"
FONT_REG = FONT_DIR / "MYRIADPRO-REGULAR.OTF"

SITE_COLORS = {
    "1P": "#F8F0C0",
    "2P": "#506868",
    "3P": "#E07878",
    "4P": "#983050",
    "5P": "#80B8A0",
}
SEASON_MARKERS = {
    "Autumn": "^",
    "Winter": "D",
    "Spring": "o",
    "Summer": "s",
}
MONTH_SEASON = {10: "Autumn", 1: "Winter", 4: "Spring", 7: "Summer"}


def load_fonts():
    if FONT_BOLD.exists() and FONT_REG.exists():
        fm.fontManager.addfont(str(FONT_BOLD))
        fm.fontManager.addfont(str(FONT_REG))
        plt.rcParams["font.family"] = "Myriad Pro"
        return (
            fm.FontProperties(fname=str(FONT_BOLD)),
            fm.FontProperties(fname=str(FONT_REG)),
        )
    plt.rcParams["font.family"] = "Arial"
    return fm.FontProperties(weight="bold"), fm.FontProperties()


font_bold, font_reg = load_fonts()


def fp(base, size):
    out = base.copy()
    out.set_size(size)
    return out


def parse_month(code) -> int:
    digits = "".join(ch for ch in str(code) if ch.isdigit())
    return int(digits[-2:])


def load_sheet2() -> pd.DataFrame:
    df = pd.read_excel(DATA, "Sheet2").rename(
        columns={"Unnamed: 0": "id", "CRAM(%)": "CRAM", "DIC(mg/L)": "DIC", "Ca (mg/L)": "Ca"}
    )
    df["site"] = df["id"].astype(str).str.split("-").str[0]
    df["season"] = df["id"].map(lambda x: MONTH_SEASON[parse_month(str(x).split("-")[-1])])
    for c in ("CRAM", "DIC", "Ca"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_c14() -> pd.DataFrame:
    df = pd.read_excel(C14).rename(
        columns={
            "SITE": "site",
            "CRAM(%)": "CRAM",
            "δ14CdIc": "d14C_DIC",
            "δ14Cdoc": "d14C_DOC",
        }
    )
    df["season"] = df["SEASON"].map(lambda x: MONTH_SEASON[parse_month(x)])
    for c in ("CRAM", "d14C_DIC", "d14C_DOC"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def format_p(p: float) -> str:
    if p < 0.01:
        return "p < 0.01"
    if p < 0.05:
        return "p < 0.05"
    return f"p = {p:.2f}"


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


def scatter_points(ax, df, x, y):
    for _, row in df.iterrows():
        ax.scatter(
            row[x], row[y],
            s=72, c=SITE_COLORS[row["site"]],
            marker=SEASON_MARKERS[row["season"]],
            edgecolors="black", linewidths=0.7, zorder=5,
        )


def panel(ax, df, x, y, xlabel, ylabel, xlim, ylim, letter):
    sub = df[[x, y, "site", "season"]].dropna()
    xv = sub[x].to_numpy(float)
    yv = sub[y].to_numpy(float)
    r, p = stats.pearsonr(xv, yv)
    xl, yl, lo, hi = fit_ci(xv, yv)
    ax.fill_between(xl, lo, hi, color="#C8C8C8", alpha=0.45, zorder=1)
    ls = "-" if p < 0.05 else "--"
    ax.plot(xl, yl, color="black", ls=ls, lw=1.7, zorder=3)
    scatter_points(ax, sub, x, y)

    ax.text(
        0.03, 0.95, f"r = {r:.2f}, {format_p(p)}",
        transform=ax.transAxes, va="top", ha="left",
        fontproperties=fp(font_bold, 11), zorder=6,
    )
    ax.set_xlabel(xlabel, fontproperties=fp(font_bold, 12))
    ax.set_ylabel(ylabel, fontproperties=fp(font_bold, 12))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=10, width=1.1, length=4)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontproperties(fp(font_reg, 10))
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.text(
        -0.14, 1.06, f"{letter})", transform=ax.transAxes,
        fontproperties=fp(font_bold, 14), va="bottom", ha="left",
    )


def legend_handles():
    sites = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SITE_COLORS[s],
               markeredgecolor="black", markeredgewidth=0.6, markersize=8, label=s)
        for s in SITE_COLORS
    ]
    seasons = [
        Line2D([0], [0], marker=SEASON_MARKERS[k], color="none", markerfacecolor="white",
               markeredgecolor="black", markeredgewidth=0.9, markersize=8, label=k)
        for k in SEASON_MARKERS
    ]
    return sites, seasons


def main() -> None:
    plt.rcParams.update({
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 1.2,
        "axes.grid": False,
    })
    env = load_sheet2()
    c14 = load_c14()

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 8.2), dpi=200)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.94, bottom=0.16, wspace=0.32, hspace=0.38)

    panel(axes[0, 0], env, "DIC", "CRAM", "DIC (mg/L)", "CRAM-like %",
          (12, 52), (35, 50), "a")
    panel(axes[0, 1], env, "Ca", "CRAM", r"Ca$^{2+}$ (mg/L)", "CRAM-like %",
          (18, 70), (35, 50), "b")
    panel(axes[1, 0], c14, "d14C_DIC", "d14C_DOC",
          r"$\Delta^{14}$C$_{\mathrm{DIC}}$ (‰)",
          r"$\Delta^{14}$C$_{\mathrm{DOC}}$ (‰)",
          (-320, -70), (-310, -90), "c")
    panel(axes[1, 1], c14, "d14C_DOC", "CRAM",
          r"$\Delta^{14}$C$_{\mathrm{DOC}}$ (‰)", "CRAM-like %",
          (-310, -90), (35, 50), "d")

    sites, seasons = legend_handles()
    leg1 = fig.legend(
        handles=sites, title="Site", loc="lower center", ncol=5,
        bbox_to_anchor=(0.32, 0.01), frameon=False, columnspacing=1.0,
        handletextpad=0.4, title_fontproperties=fp(font_bold, 11),
        prop=fp(font_reg, 10),
    )
    fig.legend(
        handles=seasons, title="Season", loc="lower center", ncol=4,
        bbox_to_anchor=(0.76, 0.01), frameon=False, columnspacing=1.0,
        handletextpad=0.4, title_fontproperties=fp(font_bold, 11),
        prop=fp(font_reg, 10),
    )
    fig.add_artist(leg1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", OUT)


if __name__ == "__main__":
    main()
