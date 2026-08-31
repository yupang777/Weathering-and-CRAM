#!/usr/bin/env python3
"""Fig. 2c — PLS-SEM path diagram.

python fig2/plot_c.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "fig2_data" / "fig2.xlsx"
OUT = ROOT / "figures" / "fig2_c.png"
MYRIAD_DIR = Path("/home/pangyu/myriad-pro")

FONT = "Myriad Pro"
FS_NODE = 13
FS_NODE_LONG = 12
FS_COEF = 11
FS_R2 = 10
FS_LEGEND = 10
PATH_LW = 2.5
NODE_HALF_W = 0.12
NODE_HALF_H = 0.07
BOX_LW = 2.0
ARROW_SCALE = 16

COEF_BBOX = dict(
    boxstyle="round,pad=0.18",
    facecolor="white",
    edgecolor="none",
    alpha=0.92,
)


def fmt_2dp(value: float) -> str:
    return f"{round(float(value), 2):.2f}"


def setup_font() -> None:
    for name in ("MYRIADPRO-REGULAR.OTF", "MYRIADPRO-BOLD.OTF"):
        path = MYRIAD_DIR / name
        if path.exists():
            fm.fontManager.addfont(str(path))
    plt.rcParams.update({
        "font.family": FONT,
        "font.sans-serif": [FONT, "Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def draw_offset_line(x0, y0, x1, y1, offset=0.0):
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length == 0:
        return x0, y0, x1, y1
    nx, ny = -dy / length, dx / length
    return (
        x0 + offset * nx,
        y0 + offset * ny,
        x1 + offset * nx,
        y1 + offset * ny,
    )


def rect_edge(cx, cy, ox, oy, hw, hh, pad=0.0):
    """Point on the rectangle around (cx, cy) facing (ox, oy)."""
    dx, dy = ox - cx, oy - cy
    if dx == 0 and dy == 0:
        return cx, cy
    tx = hw / abs(dx) if dx else math.inf
    ty = hh / abs(dy) if dy else math.inf
    t = min(tx, ty)
    length = math.hypot(dx, dy)
    return cx + t * dx + pad * dx / length, cy + t * dy + pad * dy / length


def clip_to_boxes(x0, y0, x1, y1, pad=0.006):
    sx, sy = rect_edge(x0, y0, x1, y1, NODE_HALF_W, NODE_HALF_H, pad)
    ex, ey = rect_edge(x1, y1, x0, y0, NODE_HALF_W, NODE_HALF_H, pad)
    return sx, sy, ex, ey


def perp_offset(x0, y0, x1, y1, side=1, dist=0.035):
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length == 0:
        return 0.0, 0.0
    return -side * dy / length * dist, side * dx / length * dist


def add_arrow(ax, coords, color, linestyle):
    patch = FancyArrowPatch(
        coords[:2],
        coords[2:],
        arrowstyle="-|>",
        mutation_scale=ARROW_SCALE,
        linewidth=PATH_LW,
        color=color,
        linestyle=linestyle,
        shrinkA=0,
        shrinkB=0,
        mutation_aspect=0.85,
        zorder=2,
        clip_on=False,
    )
    ax.add_patch(patch)


def draw_box(ax, x, y, label, facecolor, fontsize):
    box = Rectangle(
        (x - NODE_HALF_W, y - NODE_HALF_H),
        2 * NODE_HALF_W,
        2 * NODE_HALF_H,
        facecolor=facecolor,
        edgecolor="black",
        linewidth=BOX_LW,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        x, y, label,
        ha="center", va="center",
        fontsize=fontsize, fontweight="bold", zorder=4,
    )


def main() -> None:
    setup_font()
    effect_compare = pd.read_excel(DATA, sheet_name="c")
    inner = pd.read_excel(DATA, sheet_name="c_r2")
    r2_map = dict(zip(inner["Node"], inner["R2"]))

    node_pos = {
        "Weathering": {"label": "Weathering", "x": 0.50, "y": 0.90, "fill": "#F3C7C5", "fs": FS_NODE},
        "Hydrochemistry": {"label": "Hydrochemistry", "x": 0.18, "y": 0.58, "fill": "#BEE5E1", "fs": FS_NODE_LONG},
        "Microbial_factors": {"label": "Biotic", "x": 0.82, "y": 0.58, "fill": "#DCEE93", "fs": FS_NODE},
        "DOM_content": {"label": "DOM content", "x": 0.30, "y": 0.18, "fill": "#C7B8E1", "fs": FS_NODE_LONG},
        "DOM_stability": {"label": "DOM stability", "x": 0.70, "y": 0.18, "fill": "#9BB1E8", "fs": FS_NODE_LONG},
    }

    label_offsets = {
        "Weathering->Hydrochemistry": (-0.03, 0.03),
        "Weathering->Microbial_factors": (0.03, 0.03),
        "Hydrochemistry->Microbial_factors": (0.00, 0.05),
        "Hydrochemistry->DOM_content": (-0.05, 0.04),
        "Microbial_factors->DOM_content": (0.04, 0.04),
        "Weathering->DOM_stability": (-0.04, 0.00),
        "Microbial_factors->DOM_stability": (0.06, 0.04),
        "DOM_content->DOM_stability": (0.00, 0.04),
        "Weathering=>DOM_content": (0.02, -0.02),
        "Hydrochemistry=>DOM_stability": (0.04, 0.06),
    }
    label_sides = {
        "Weathering->Hydrochemistry": 1,
        "Weathering->Microbial_factors": -1,
        "Hydrochemistry->Microbial_factors": -1,
        "Hydrochemistry->DOM_content": 1,
        "Microbial_factors->DOM_content": -1,
        "Weathering->DOM_stability": 1,
        "Microbial_factors->DOM_stability": -1,
        "DOM_content->DOM_stability": 1,
        "Weathering=>DOM_content": -1,
        "Hydrochemistry=>DOM_stability": 1,
    }
    label_perp_dist = {
        "Hydrochemistry->Microbial_factors": 0.050,
        "Hydrochemistry->DOM_content": 0.055,
        "Microbial_factors->DOM_content": 0.050,
        "Microbial_factors->DOM_stability": 0.055,
        "Weathering=>DOM_content": 0.050,
        "Hydrochemistry=>DOM_stability": 0.050,
    }
    r2_offsets = {
        "Hydrochemistry": (-0.02, -0.115),
        "Microbial_factors": (0.03, -0.115),
        "DOM_content": (-0.02, -0.115),
        "DOM_stability": (0.02, -0.115),
    }

    plot_df = effect_compare.dropna(subset=["chosen_type", "chosen_est"]).copy()
    rel_split = plot_df["relationship"].str.split(" -> ", n=1, expand=True)
    plot_df["from"] = rel_split[0]
    plot_df["to"] = rel_split[1]

    fig, ax = plt.subplots(figsize=(10.9, 7.7), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.01, 0.98, "c)", fontsize=16, fontweight="bold", va="top")

    for _, row in plot_df.iterrows():
        src = node_pos[row["from"]]
        dst = node_pos[row["to"]]
        est = float(row["chosen_est"])
        color = "#1F77B4" if est >= 0 else "black"
        linestyle = "-" if row["chosen_type"] == "direct" else "--"
        offset_amt = 0.0 if row["chosen_type"] == "direct" else 0.015
        mid = draw_offset_line(src["x"], src["y"], dst["x"], dst["y"], offset=offset_amt)
        coords = clip_to_boxes(*mid)
        add_arrow(ax, coords, color=color, linestyle=linestyle)

        key = (
            f"{row['from']}->{row['to']}"
            if row["chosen_type"] == "direct"
            else f"{row['from']}=>{row['to']}"
        )
        ox, oy = label_offsets.get(key, (0.0, 0.03))
        side = label_sides.get(key, 1)
        perp = label_perp_dist.get(key, 0.035)
        px, py = perp_offset(*coords, side=side, dist=perp)
        tx = (coords[0] + coords[2]) / 2 + ox + px
        ty = (coords[1] + coords[3]) / 2 + oy + py
        stars = row["chosen_stars"] if isinstance(row["chosen_stars"], str) else ""
        ax.text(
            tx, ty, f"{fmt_2dp(est)}{stars}",
            ha="center", va="center",
            fontsize=FS_COEF, fontweight="bold", color=color,
            bbox=COEF_BBOX, zorder=5,
        )

    for node, info in node_pos.items():
        draw_box(ax, info["x"], info["y"], info["label"], info["fill"], info["fs"])
        r2_val = r2_map.get(node)
        if pd.notna(r2_val) and r2_val > 0:
            ox, oy = r2_offsets.get(node, (0.0, -0.115))
            ax.text(
                info["x"] + ox, info["y"] + oy,
                rf"$R^2$ = {fmt_2dp(r2_val)}",
                ha="center", va="center", fontsize=FS_R2, zorder=5,
            )

    legend_items = [
        ("Direct", "black", "-"),
        ("Indirect", "black", "--"),
        ("Positive", "#1F77B4", "-"),
        ("Negative", "black", "-"),
    ]
    for i, (text, color, linestyle) in enumerate(legend_items):
        y = 0.97 - i * 0.035
        ax.plot([0.84, 0.885], [y, y], color=color, linestyle=linestyle, linewidth=PATH_LW)
        ax.text(0.895, y, text, ha="left", va="center", fontsize=FS_LEGEND)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", OUT)


if __name__ == "__main__":
    main()
