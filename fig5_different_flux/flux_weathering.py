#!/usr/bin/env python3
"""CRAM already produced by weathering.

F_weath = F_global * f_auto
f_auto = mean(Auto%/100) at Puding (n=20)
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
LIU = 0.30

curves = pd.read_excel(ROOT / "fig4" / "fig4.xlsx", sheet_name="d")
puding = pd.read_excel(HERE / "data" / "puding.xlsx")
c14 = pd.read_excel(ROOT / "fig3" / "fig3_data.xlsx")

now = curves[curves["sensitivity_label"] == "1.0x original saturation"].iloc[0]
F_global = LIU * float(now["mean_cram_pct"]) / 100.0
f_auto = float(puding["Auto"].mean() / 100.0)
F_weath = F_global * f_auto

doc = c14["δ14Cdoc"].to_numpy(float)
dic = c14["δ14CdIc"].to_numpy(float)
den = dic - 0.0
f_raw = np.where(np.abs(den) > 1e-9, (doc - 0.0) / den, np.nan)
f_c14 = float(np.nanmean(np.clip(f_raw, 0, 1)))

out = pd.DataFrame([
    {
        "quantity": "F_weath",
        "pg_c_yr": F_weath,
        "formula": "F_global * f_auto",
        "note": f"f_auto = {f_auto:.3f}  (Puding Auto n=20)",
    },
    {
        "quantity": "F_weath_14C",
        "pg_c_yr": F_global * f_c14,
        "formula": "F_global * f_14C",
        "note": f"f_14C = {f_c14:.3f}  (Δ14C other=0‰, clipped, n=10)",
    },
])
path = HERE / "data" / "flux_weathering.csv"
path.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(path, index=False)
print(f"F_weath   {F_weath:.4f} Pg C/yr   F_global × f_auto {f_auto:.3f}")
print(f"F_weath_14C  {F_global * f_c14:.4f} Pg C/yr   F_global × f_14C {f_c14:.3f}")
print("saved", path)
