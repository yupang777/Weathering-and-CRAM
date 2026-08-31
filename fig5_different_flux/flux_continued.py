#!/usr/bin/env python3
"""Additional CRAM if weathering continues to saturation.

F_more = 0.30 * (CRAM_sat% - CRAM_now%) / 100
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
LIU = 0.30

curves = pd.read_excel(ROOT / "fig4" / "fig4.xlsx", sheet_name="d")
rows = []
for _, g in curves.groupby("sensitivity_label", sort=False):
    a, b = g.iloc[0], g.iloc[-1]
    cram_now = float(a["mean_cram_pct"])
    cram_sat = float(b["mean_cram_pct"])
    m = float(a["saturation_gap_multiplier"])
    F_more = LIU * (cram_sat - cram_now) / 100.0
    rows.append({
        "quantity": "F_more" if m == 1.0 else f"F_more_{m:.1f}x",
        "pg_c_yr": F_more,
        "formula": "0.30 * (CRAM_sat% - CRAM_now%) / 100",
        "note": f"{a['sensitivity_label']}: {cram_now:.2f}% → {cram_sat:.2f}%",
    })
    if m == 1.0:
        print(f"F_more    {F_more:.4f} Pg C/yr   0.30 × ({cram_sat:.2f} − {cram_now:.2f})%")

out = pd.DataFrame(rows)
path = HERE / "data" / "flux_continued.csv"
path.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(path, index=False)
print("saved", path)
