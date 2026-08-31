#!/usr/bin/env python3
"""Fig. 1 a — global site map.

python fig1/plot_map.py
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "global_pearl_puding data.xlsx"
CARBONATE = ROOT / "gis" / "carbonate" / "Carbonate_rock_distribution.shp"
OUT = ROOT / "figures" / "fig1_map.png"

PUDING = (105.75, 26.32)
COLOR_CARB = "#6BA3C7"
COLOR_LAND = "#F4F1EA"
COLOR_GLOBAL = "#8B1E3F"
COLOR_PEARL = "#C0392B"
COLOR_PUDING = "#E67E22"


def main() -> None:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    pearl = pd.read_excel(DATA, "pearl river")
    glob = pd.read_excel(DATA, "global")
    carb = gpd.read_file(CARBONATE)
    carb = carb[carb["rock_type"].isin([1, 2, 4])]

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Myriad Pro", "Arial"]})
    fig = plt.figure(figsize=(10.5, 5.4), dpi=200)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([-170, 180, -55, 80], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor=COLOR_LAND, edgecolor="none")
    ax.add_feature(cfeature.OCEAN, facecolor="white", edgecolor="none")
    ax.coastlines(linewidth=0.4, color="#666666")
    ax.add_geometries(
        carb.geometry, crs=ccrs.PlateCarree(),
        facecolor=COLOR_CARB, edgecolor="none", alpha=0.85, zorder=1,
    )
    ax.scatter(
        glob["lon"], glob["lat"], s=14, c=COLOR_GLOBAL, zorder=4,
        transform=ccrs.PlateCarree(), edgecolors="none",
    )
    ax.scatter(
        pearl["lon"], pearl["lat"], s=7, c=COLOR_PEARL, zorder=5,
        transform=ccrs.PlateCarree(), edgecolors="none",
    )
    ax.scatter(
        [PUDING[0]], [PUDING[1]], s=80, marker="*", c=COLOR_PUDING,
        edgecolors="white", linewidths=0.4, zorder=6, transform=ccrs.PlateCarree(),
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.0, 1.03, "a)", transform=ax.transAxes, fontweight="bold",
            fontsize=14, va="bottom", ha="left")
    ax.legend(
        handles=[
            mpatches.Patch(facecolor=COLOR_CARB, label="Carbonate rocks"),
            mpatches.Patch(facecolor=COLOR_LAND, edgecolor="#888", label="Non-carbonate rocks"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_GLOBAL,
                   markersize=7, label="Global sampling sites"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_PEARL,
                   markersize=6, label="Pearl River sites"),
            Line2D([0], [0], marker="*", color="none", markerfacecolor=COLOR_PUDING,
                   markersize=11, label="Puding Station"),
        ],
        loc="lower left", frameon=False, fontsize=8,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", OUT)


if __name__ == "__main__":
    main()
