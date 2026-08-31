#!/usr/bin/env python3
"""Fig. 2a — Spearman bubble plots.

python fig2/plot_a.py
"""

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = Path("/home/pangyu/myriad-pro")
FONT_BOLD = FONT_DIR / "MYRIADPRO-BOLD.OTF"
FONT_REG = FONT_DIR / "MYRIADPRO-REGULAR.OTF"
DATA = ROOT / "fig2_data" / "fig2.xlsx"
OUT_DIR = ROOT / "figures"

FACE = {
    "Hydrochemistry": "#E8F4F2",
    "Weathering": "#F6E8E8",
    "Biotic": "#F3F6DC",
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


font_bold, font_reg = load_fonts()


def fp(base, size):
    out = base.copy()
    out.set_size(size)
    return out


def get_star(p):
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def plot_bubble_heatmap(ax, row_vars, col_vars, df, rotation=45,
                        bubble_scale=3200, star_size=22, label_size=16):
    n_rows, n_cols = len(row_vars), len(col_vars)
    x, y = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
    x, y = x.flatten(), y.flatten()

    s_vals, c_vals, stars = [], [], []
    for r_var in row_vars:
        for c_var in col_vars:
            if r_var not in df.columns or c_var not in df.columns:
                s_vals.append(0)
                c_vals.append(0)
                stars.append("")
                continue
            valid = df[r_var].notna() & df[c_var].notna()
            if valid.sum() < 2:
                corr, p = 0.0, 1.0
            else:
                corr, p = spearmanr(df.loc[valid, r_var], df.loc[valid, c_var])
            s_vals.append(abs(corr) * bubble_scale)
            c_vals.append(corr)
            stars.append(get_star(p))

    y_plot = (n_rows - 1) - y
    sc = ax.scatter(
        x, y_plot, s=s_vals, c=c_vals, cmap="RdBu_r",
        vmin=-1, vmax=1, edgecolors="none", zorder=3,
    )
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5)

    for i, star in enumerate(stars):
        if star:
            ax.text(
                x[i], y_plot[i], star, ha="center", va="center",
                fontsize=star_size, color="black", fontproperties=font_reg, zorder=4,
            )

    label_font = fp(font_bold, label_size)
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(col_vars, fontproperties=label_font)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_vars[::-1], fontproperties=label_font)
    ax.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_rows + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="#CCCCCC", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", size=0)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.8)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    plt.setp(ax.get_xticklabels(), rotation=rotation, ha="left", rotation_mode="anchor")
    return sc


def save_one(df, row_vars, col_vars, group_name, filename,
             inner_width, inner_height, rotation=45):
    left_margin, right_margin, top_margin, bottom_margin = 3.2, 1.6, 2.4, 0.4
    fig_w = left_margin + inner_width + right_margin
    fig_h = top_margin + inner_height + bottom_margin
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([
        left_margin / fig_w,
        bottom_margin / fig_h,
        inner_width / fig_w,
        inner_height / fig_h,
    ])
    ax.set_facecolor(FACE[group_name])
    sc = plot_bubble_heatmap(
        ax, row_vars, col_vars, df, rotation=rotation,
        bubble_scale=2800, star_size=16, label_size=13,
    )
    ax.set_xlabel(group_name, labelpad=14, fontproperties=fp(font_bold, 18))
    ax.xaxis.set_label_position("top")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.35)
    cbar = plt.colorbar(sc, cax=cax)
    cbar.set_label("Correlation index", fontproperties=fp(font_reg, 13))
    cbar.ax.tick_params(labelsize=11)
    for lab in cbar.ax.get_yticklabels():
        lab.set_fontproperties(fp(font_reg, 11))
    cbar.outline.set_linewidth(1.0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=300, facecolor="white")
    plt.close(fig)
    print("saved", filename)


def load_table():
    return pd.read_excel(DATA, sheet_name="a", header=1)


def rename_columns(df):
    new_names = {
        0: "DOC",
        1: "Autochthonous DOM",
        2: "Allochthonous DOM",
        3: "Auto-S",
        4: "Allo-S",
        5: "CRAM-like %",
        7: "WT",
        8: r"$\mathbf{DO_{sat}}$",
        9: "pH",
        10: r"$\mathbf{\delta^{2}H}$",
        11: "TN",
        12: "TP",
        13: "Con",
        14: r"$\mathbf{F^{-}}$",
        15: r"$\mathbf{Cl^{-}}$",
        16: r"$\mathbf{SO_{4}^{2-}}$",
        17: "DIC",
        18: r"$\mathbf{Ca^{2+}}$",
        19: r"$\mathbf{Mg^{2+}}$",
        20: r"$\mathbf{K^{+}}$",
        21: r"$\mathbf{Na^{+}}$",
        22: r"$\mathbf{Si^{4+}}$",
        23: r"$\mathbf{Sr^{2+}}$",
        24: r"$\mathbf{Fe^{3+}}$",
        25: r"$\mathbf{Mn^{2+}}$",
        26: r"$\mathbf{Zn^{2+}}$",
        27: r"$\mathbf{Ba^{2+}}$",
        28: "Fe/Mn",
    }
    for i, col in enumerate(df.columns[29:]):
        name = col.replace("_", " ").strip()
        name = name[0].upper() + name[1:]
        low = col.lower()
        if "aromatic" in low:
            name = "Aromatic compound\ndegradation"
        elif "aerobic" in low and "chemo" in low:
            name = "Aerobic\nchemoheterotrophy"
        elif "dark" in low and "hydrogen" in low:
            name = "Dark hydrogen\noxidation"
        elif "dark" in low and "sulfur" in low:
            name = "Dark sulfur\noxidation"
        elif "dark" in low and "thiosulfate" in low:
            name = "Dark thiosulfate\noxidation"
        elif "oxygenic" in low:
            name = "Oxygenic\nphotoautotrophy"
        elif "photosynthetic" in low:
            name = "Photosynthetic\ncyanobacteria"
        new_names[29 + i] = name

    cols = df.columns.tolist()
    df = df.rename(columns={cols[i]: name for i, name in new_names.items() if i < len(cols)})
    return df


def main():
    plt.rcParams.update({
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    df = rename_columns(load_table())
    r1_vars = [df.columns[i] for i in range(6)]
    hydro_vars = df.columns[7:17].tolist()
    weather_vars = df.columns[17:29].tolist()
    biotic_vars = df.columns[29:].tolist()

    col_w, row_h = 1.15, 0.48
    inner_w = len(r1_vars) * col_w

    save_one(
        df, hydro_vars, r1_vars, "Hydrochemistry",
        OUT_DIR / "fig2_a_hydro.png",
        inner_w, len(hydro_vars) * row_h,
    )
    save_one(
        df, weather_vars, r1_vars, "Weathering",
        OUT_DIR / "fig2_a_weathering.png",
        inner_w, len(weather_vars) * row_h,
    )
    save_one(
        df, biotic_vars, r1_vars, "Biotic",
        OUT_DIR / "fig2_a_biotic.png",
        inner_w, max(len(biotic_vars) * 0.42, 8.0),
        rotation=45,
    )


if __name__ == "__main__":
    main()
