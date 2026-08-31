#!/usr/bin/env python3
"""Fig. 2b — Random Forest %IncMSE.

python fig2/plot_b.py
"""

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = Path("/home/pangyu/myriad-pro")
FONT_BOLD = FONT_DIR / "MYRIADPRO-BOLD.OTF"
FONT_REG = FONT_DIR / "MYRIADPRO-REGULAR.OTF"
DATA = ROOT / "fig2_data" / "fig2.xlsx"
OUT = ROOT / "figures" / "fig2_b.png"

SEM_COLORS = {
    "Weathering": "#E6C2C2",
    "Hydrochemistry": "#BCE2D8",
    "Biotic": "#D5E490",
}

LABEL = {
    "Ca2+": r"Ca$^{\mathbf{2+}}$",
    "Sr2+": r"Sr$^{\mathbf{2+}}$",
    "F-": r"F$^{\mathbf{-}}$",
    "d2H": r"$\mathbf{\delta^{2}}$H",
    "Ba2+": r"Ba$^{\mathbf{2+}}$",
}


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


def main() -> None:
    font_bold, font_reg = load_fonts()
    plt.rcParams.update({
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    df = pd.read_excel(DATA, sheet_name="b")
    df["Stars"] = df["Stars"].fillna("").astype(str).replace("nan", "")
    df["Feature"] = df["Feature"].map(lambda x: LABEL.get(str(x), str(x)))
    df = df.iloc[::-1].reset_index(drop=True)
    colors = [SEM_COLORS[c] for c in df["Category"]]

    fig, ax = plt.subplots(figsize=(13, 11))
    plt.subplots_adjust(left=0.42, right=0.95, top=0.95, bottom=0.10)

    ax.barh(
        df["Feature"], df["IncMSE"],
        color=colors, edgecolor="black", linewidth=1.5, height=0.7,
    )

    font_xlabel = font_bold.copy()
    font_xlabel.set_size(28)
    font_tick = font_bold.copy()
    font_tick.set_size(24)
    font_star = font_bold.copy()
    font_star.set_size(22)
    font_letter = font_bold.copy()
    font_letter.set_size(28)
    font_leg = font_reg.copy()
    font_leg.set_size(20)

    ax.set_xlabel("Increase in MSE (%)", fontproperties=font_xlabel, labelpad=15)
    ax.set_xticks(range(8))
    ax.set_xticklabels(range(8), fontproperties=font_tick)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["Feature"], fontproperties=font_tick)
    ax.set_xlim(0, 7.5)
    ax.tick_params(axis="both", direction="in", length=5, width=1.2)

    for i, row in df.iterrows():
        if row["Stars"]:
            ax.text(
                row["IncMSE"] + 0.12, i, row["Stars"],
                va="center", ha="left", fontproperties=font_star, color="black",
            )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.0)
        spine.set_color("black")

    ax.legend(
        handles=[
            Patch(facecolor=SEM_COLORS["Weathering"], edgecolor="black", label="Weathering"),
            Patch(facecolor=SEM_COLORS["Hydrochemistry"], edgecolor="black", label="Hydrochemistry"),
            Patch(facecolor=SEM_COLORS["Biotic"], edgecolor="black", label="Biotic"),
        ],
        loc="lower right", frameon=False, prop=font_leg,
    )
    ax.text(0.02, 1.02, "b)", transform=ax.transAxes, fontproperties=font_letter, va="bottom")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", OUT)


if __name__ == "__main__":
    main()
