#!/usr/bin/env python3
"""Fig. 1 b–c — DIC / CRAM violins.

python fig1/plot_bc.py
"""

from pathlib import Path

import matplotlib.cm as cm
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from statannotations.Annotator import Annotator

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = Path("/home/pangyu/myriad-pro")
FONT_BOLD = FONT_DIR / "MYRIADPRO-BOLD.OTF"
FONT_REG = FONT_DIR / "MYRIADPRO-REGULAR.OTF"
DATA_XLSX = ROOT / "data" / "global_pearl_puding data.xlsx"
STATS_XLSX = ROOT / "data" / "basin_stats.xlsx"
OUT = ROOT / "figures" / "fig1_bc.png"

colors_purple_distinct = ["#F3EBF5", "#E8DAEB", "#CDA3CB", "#B266A5", "#8E4483"]
cmap_final = mcolors.LinearSegmentedColormap.from_list(
    "custom_purple_distinct", colors_purple_distinct
)


def load_fonts():
    try:
        fm.fontManager.addfont(str(FONT_BOLD))
        fm.fontManager.addfont(str(FONT_REG))
        font_reg = fm.FontProperties(fname=str(FONT_REG))
        plt.rcParams["font.family"] = font_reg.get_name()
        return True
    except Exception:
        plt.rcParams["font.family"] = "Arial"
        return False


def fp(path, size, weight=None):
    if path.exists():
        return fm.FontProperties(fname=str(path), size=size)
    return fm.FontProperties(size=size, weight=weight or "normal")


def draw_panel(ax, df, y_col, ylabel, order, palette, pairs, letter):
    sns.violinplot(
        x="Zone_Mapped", y=y_col, data=df, order=order,
        palette=palette, ax=ax, saturation=1.0,
        inner=None, linewidth=1.5, density_norm="width", cut=2,
    )
    for collection in ax.collections:
        if isinstance(collection, mcoll.PolyCollection):
            collection.set_alpha(0.8)

    sns.boxplot(
        x="Zone_Mapped", y=y_col, data=df, order=order,
        ax=ax, width=0.15, color="white",
        boxprops={"zorder": 2, "edgecolor": "#333333", "linewidth": 2.0, "alpha": 0.95},
        whiskerprops={"zorder": 2, "color": "#333333", "linewidth": 2.0},
        medianprops={"color": "#333333", "linewidth": 2.5},
        showfliers=False,
    )
    sns.stripplot(
        x="Zone_Mapped", y=y_col, data=df, order=order,
        ax=ax, jitter=0.15, size=6, color="#404040", alpha=0.5, zorder=3,
    )

    p_values = [
        stats.ttest_ind(
            df[df["Zone_Mapped"] == a][y_col].dropna(),
            df[df["Zone_Mapped"] == b][y_col].dropna(),
        )[1]
        for a, b in pairs
    ]
    sig_pairs = [pair for pair, p in zip(pairs, p_values) if p <= 0.05]
    sig_p = [p for p in p_values if p <= 0.05]
    if sig_pairs:
        annotator = Annotator(ax, sig_pairs, data=df, x="Zone_Mapped", y=y_col, order=order)
        annotator.configure(
            test=None, text_format="star", loc="inside",
            verbose=0, line_height=0.015, text_offset=8, line_width=2.0, fontsize=22,
        )
        annotator.set_pvalues(sig_p)
        annotator.annotate()

    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontproperties=fp(FONT_BOLD, 18, "bold"))
    if "DIC" in y_col:
        ax.set_ylim(bottom=0)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(
        order, rotation=45, ha="right",
        fontproperties=fp(FONT_BOLD, 14, "bold"), rotation_mode="anchor",
    )
    ax.tick_params(axis="y", labelsize=13)
    for label in ax.get_yticklabels():
        label.set_fontproperties(fp(FONT_REG, 13))
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_color("black")
    ax.text(-0.14, 1.04, f"{letter})", transform=ax.transAxes,
            fontproperties=fp(FONT_BOLD, 16, "bold"), va="bottom")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.08)
    sm = cm.ScalarMappable(cmap=cmap_final, norm=mcolors.Normalize(
        vmin=df["Carbonate"].min(), vmax=df["Carbonate"].max()
    ))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label("Carbonate Rock (%)", fontproperties=fp(FONT_BOLD, 10, "bold"))
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(fp(FONT_REG, 9))


def main():
    load_fonts()
    df_raw = pd.read_excel(DATA_XLSX, sheet_name="pearl river")
    df_raw["Zone_Mapped"] = df_raw["Zone"]
    df_stats = pd.read_excel(STATS_XLSX)
    df_stats["Sub-basin"] = df_stats["Sub-basin"].replace({"NJ(NBP)": "NBPJ", "HJ(HL)": "HLJ"})
    df = pd.merge(
        df_raw, df_stats[["Sub-basin", "Carbonate"]],
        left_on="Zone_Mapped", right_on="Sub-basin", how="left",
    )
    order = ["NBPJ", "HLJ", "YJ", "XJ", "BJ", "DJ", "PRD"]
    norm = mcolors.Normalize(vmin=df["Carbonate"].min(), vmax=df["Carbonate"].max())
    palette = {row["Zone_Mapped"]: cmap_final(norm(row["Carbonate"])) for _, row in df.iterrows()}
    pairs = [(order[i], order[i + 1]) for i in range(len(order) - 1)]
    pairs.append((order[0], order[-1]))

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 10.5))
    plt.subplots_adjust(hspace=0.32)
    draw_panel(axes[0], df, "DIC(mg/L)", "DIC (mg/L)", order, palette, pairs, "b")
    draw_panel(axes[1], df, "CRAM(%)", "CRAM-like %", order, palette, pairs, "c")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", OUT)


if __name__ == "__main__":
    main()
