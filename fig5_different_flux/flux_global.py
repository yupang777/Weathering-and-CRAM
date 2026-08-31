#!/usr/bin/env python3
"""Contemporary global CRAM flux.

F_global = 0.30 * CRAM_now% / 100
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
LIU = 0.30

curves = pd.read_excel(ROOT / "fig4" / "fig4.xlsx", sheet_name="d")
now = curves[curves["sensitivity_label"] == "1.0x original saturation"].iloc[0]
cram_now = float(now["mean_cram_pct"])
F_global = LIU * cram_now / 100.0

out = pd.DataFrame([{
    "quantity": "F_global",
    "pg_c_yr": F_global,
    "formula": "0.30 * CRAM_now% / 100",
    "note": f"CRAM_now = {cram_now:.2f}%  (progress {float(now['progress_pct']):.1f}%)",
}])
path = HERE / "data" / "flux_global.csv"
path.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(path, index=False)
print(f"F_global  {F_global:.4f} Pg C/yr   Liu 0.30 × {cram_now:.2f}%")
print("saved", path)
