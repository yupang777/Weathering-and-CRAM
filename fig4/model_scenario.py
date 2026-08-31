#!/usr/bin/env python3
"""HistGB weathering scenarios for Fig. 4 c–d. Writes fig4.xlsx sheets c and d."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
XLSX = HERE / "fig4.xlsx"
SAT_TIF = HERE / "DIC_mmol_L.tif"
SAT_TIF_LOCAL = Path("/home/pangyu/aoe/ions/2026-4-22/DIC_mmol_L.tif")

DIC_C, CA_MG = 12.01, 40.078
K, N_MC, LIU = 3.0, 1000, 0.30
GRID = np.linspace(0, 100, 51)
GAPS = {
    "1.0x original saturation": 1.0,
    "1.1x enhanced gap": 1.1,
    "1.3x enhanced gap": 1.3,
    "1.5x enhanced gap": 1.5,
}
SMOOTH = [
    "mean_cram_pct", "mean_delta_cram_pct",
    "mc_p025_cram_pct", "mc_p975_cram_pct",
    "mc_p025_delta_cram_pct", "mc_p975_delta_cram_pct",
]


def sat_tif(path=None):
    for p in (path, SAT_TIF, SAT_TIF_LOCAL):
        if p is not None and Path(p).exists():
            return Path(p)
    raise FileNotFoundError("pass --sat-tif; raster is not in the repo")


def read_sites():
    df = pd.read_excel(XLSX, sheet_name="train").rename(columns={
        "Sites": "sample", "Zone": "zone", "CRAM(%)": "CRAM_obs_pct",
        "DIC(mg/L)": "DIC_obs_mg_L", "Ca2+(mg/L)": "Ca_obs_mg_L",
    })
    for c in ("lon", "lat", "CRAM_obs_pct", "DIC_obs_mg_L", "Ca_obs_mg_L"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def extract_sat(df, tif):
    import rasterio
    vals = []
    with rasterio.open(tif) as src:
        b = src.bounds
        for lon, lat in zip(df["lon"], df["lat"]):
            if pd.isna(lon) or pd.isna(lat) or not (b.left <= lon <= b.right and b.bottom <= lat <= b.top):
                vals.append(np.nan)
                continue
            v = next(src.sample([(float(lon), float(lat))]))[0]
            vals.append(float(v) if np.isfinite(v) else np.nan)
    out = df.copy()
    out["DIC_sat_mg_L_raw"] = np.array(vals) * DIC_C
    out["Ca_sat_mg_L_raw"] = np.array(vals) / 2.0 * CA_MG
    return out


def apply_gap(df, m):
    df = df.copy()
    df["saturation_gap_multiplier"] = m
    df["DIC_sat_mg_L"] = df["DIC_obs_mg_L"] + m * df["DIC_gap_raw_mg_L"]
    df["Ca_sat_mg_L"] = df["Ca_obs_mg_L"] + m * df["Ca_gap_raw_mg_L"]
    df["DIC_gap_mg_L"] = df["DIC_sat_mg_L"] - df["DIC_obs_mg_L"]
    df["Ca_gap_mg_L"] = df["Ca_sat_mg_L"] - df["Ca_obs_mg_L"]
    wi_obs = np.hypot(
        np.log1p(df["DIC_obs_mg_L"].clip(lower=0) / DIC_C),
        np.log1p(df["Ca_obs_mg_L"].clip(lower=0) / CA_MG),
    )
    wi_sat = np.hypot(
        np.log1p(df["DIC_sat_mg_L"].clip(lower=0) / DIC_C),
        np.log1p(df["Ca_sat_mg_L"].clip(lower=0) / CA_MG),
    )
    prog = (wi_obs / wi_sat.replace(0, np.nan) * 100).clip(0, 100)
    if "current_progress_reference_pct" in df.columns:
        df["current_progress_pct"] = df["current_progress_reference_pct"]
    else:
        df["current_progress_pct"] = prog
        df["current_progress_reference_pct"] = prog
    return df.dropna(subset=["current_progress_pct"]).reset_index(drop=True)


def prepare(tif):
    df = extract_sat(read_sites(), tif)
    df = df.dropna(subset=["CRAM_obs_pct", "DIC_obs_mg_L", "Ca_obs_mg_L", "DIC_sat_mg_L_raw", "Ca_sat_mg_L_raw"])
    df["zone"] = df["zone"].fillna("Unknown").astype(str)
    df["DIC_obs_mg_L"] = df["DIC_obs_mg_L"].clip(lower=0)
    df["Ca_obs_mg_L"] = df["Ca_obs_mg_L"].clip(lower=0)
    df["DIC_gap_raw_mg_L"] = (df["DIC_sat_mg_L_raw"] - df["DIC_obs_mg_L"]).clip(lower=0)
    df["Ca_gap_raw_mg_L"] = (df["Ca_sat_mg_L_raw"] - df["Ca_obs_mg_L"]).clip(lower=0)
    return apply_gap(df, 1.0)


class Model:
    def __init__(self, seed=42):
        self.seed = seed
        self.zone_cols = []
        self.est = None

    def _zones(self, zone, n, fit=False):
        s = pd.Series(np.asarray(zone).ravel()).fillna("Unknown").astype(str)
        if fit:
            self.zone_cols = sorted(f"zone_{z}" for z in s.unique())
        out = pd.DataFrame(0.0, index=np.arange(n), columns=self.zone_cols)
        for z in s.unique():
            col = f"zone_{z}"
            if col in out.columns:
                out.loc[s == z, col] = 1.0
        return out

    def _X(self, dic, ca, zone, fit=False):
        dic = np.clip(pd.to_numeric(dic, errors="coerce"), 0, None)
        ca = np.clip(pd.to_numeric(ca, errors="coerce"), 0, None)
        x = pd.DataFrame({
            "log_dic": np.log1p(dic / DIC_C),
            "log_ca": np.log1p(ca / CA_MG),
        }).reset_index(drop=True)
        return pd.concat([x, self._zones(zone, len(x), fit=fit)], axis=1)

    def fit(self, df):
        X = self._X(df["DIC_obs_mg_L"], df["Ca_obs_mg_L"], df["zone"], fit=True)
        self.est = HistGradientBoostingRegressor(
            max_iter=380, learning_rate=0.03, max_leaf_nodes=31,
            l2_regularization=0.0, min_samples_leaf=5,
            monotonic_cst=[1, 1] + [0] * (X.shape[1] - 2),
            random_state=self.seed,
        )
        self.est.fit(X, df["CRAM_obs_pct"])
        return self

    def predict(self, dic, ca, zone):
        return self.est.predict(self._X(dic, ca, zone))


def curve(df, model, label):
    base = model.predict(df["DIC_obs_mg_L"], df["Ca_obs_mg_L"], df["zone"])
    t = GRID / 100.0
    frac = (1 - np.exp(-K * t)) / (1 - np.exp(-K))
    med = df["current_progress_pct"].median()
    progress = med + t * (100 - med)
    n = len(df)
    dic = np.tile(df["DIC_obs_mg_L"].to_numpy(), (len(GRID), 1)) + np.outer(frac, df["DIC_gap_mg_L"])
    ca = np.tile(df["Ca_obs_mg_L"].to_numpy(), (len(GRID), 1)) + np.outer(frac, df["Ca_gap_mg_L"])
    zone = np.tile(df["zone"].to_numpy(), (len(GRID), 1))
    pred = model.predict(dic.ravel(), ca.ravel(), zone.ravel()).reshape(len(GRID), n)
    return pd.DataFrame({
        "sensitivity_label": label,
        "saturation_gap_multiplier": float(df["saturation_gap_multiplier"].iloc[0]),
        "progress_pct": progress,
        "mean_cram_pct": pred.mean(axis=1),
        "mean_delta_cram_pct": (pred - base.reshape(1, -1)).mean(axis=1),
    })


def monte_carlo(df, n_mc=N_MC):
    rng = np.random.default_rng(42)
    base_model = Model(42).fit(df)
    rows, sites = [], []
    for label, m in GAPS.items():
        sub = apply_gap(df, m)
        rows.append(curve(sub, base_model, label))
        ep = sub.copy()
        ep["CRAM_baseline_pred_pct"] = base_model.predict(ep["DIC_obs_mg_L"], ep["Ca_obs_mg_L"], ep["zone"])
        ep["sensitivity_label"] = label
        sites.append(ep)
    curves = pd.concat(rows, ignore_index=True)
    site_results = pd.concat(sites, ignore_index=True)
    mc = []
    for i in range(n_mc):
        if i % 50 == 0:
            print(f"Monte Carlo {i + 1}/{n_mc}")
        boot = df.iloc[rng.integers(0, len(df), len(df))].copy()
        model = Model(43 + i).fit(boot)
        one = pd.concat([curve(apply_gap(df, m), model, lab) for lab, m in GAPS.items()], ignore_index=True)
        one["mc_id"] = i + 1
        mc.append(one)
    ci = (
        pd.concat(mc, ignore_index=True)
        .groupby(["sensitivity_label", "saturation_gap_multiplier", "progress_pct"])
        .agg(
            mc_p025_cram_pct=("mean_cram_pct", lambda x: np.percentile(x, 2.5)),
            mc_p975_cram_pct=("mean_cram_pct", lambda x: np.percentile(x, 97.5)),
            mc_p025_delta_cram_pct=("mean_delta_cram_pct", lambda x: np.percentile(x, 2.5)),
            mc_p975_delta_cram_pct=("mean_delta_cram_pct", lambda x: np.percentile(x, 97.5)),
        )
        .reset_index()
    )
    return curves.merge(ci, on=["sensitivity_label", "saturation_gap_multiplier", "progress_pct"]), site_results


def smooth(curves):
    out = curves.copy()
    for _, idx in out.groupby("sensitivity_label").groups.items():
        sub = out.loc[idx].sort_values("progress_pct")
        for col in SMOOTH:
            if col in out.columns:
                out.loc[sub.index, col] = pd.Series(sub[col]).rolling(3, center=True, min_periods=1).mean().to_numpy()
    return out


def scale_liu(curves):
    out = curves.copy()
    out["global_doc_flux_pg_yr"] = LIU
    out["cram_flux_pg_yr"] = LIU * out["mean_cram_pct"] / 100
    out["cram_flux_delta_pg_yr"] = LIU * out["mean_delta_cram_pct"] / 100
    out["cram_flux_pg_yr_mc_lo"] = LIU * out["mc_p025_cram_pct"] / 100
    out["cram_flux_pg_yr_mc_hi"] = LIU * out["mc_p975_cram_pct"] / 100
    out["cram_flux_delta_pg_yr_mc_lo"] = LIU * out["mc_p025_delta_cram_pct"] / 100
    out["cram_flux_delta_pg_yr_mc_hi"] = LIU * out["mc_p975_delta_cram_pct"] / 100
    return out


def write_cd(curves, progress):
    curves = scale_liu(curves)
    for col in ("median_progress_pct", "q25_progress_pct", "q75_progress_pct"):
        curves[col] = float(progress[col])
    sheet_c = curves[[
        "sensitivity_label", "saturation_gap_multiplier", "progress_pct",
        "cram_flux_delta_pg_yr", "cram_flux_delta_pg_yr_mc_lo", "cram_flux_delta_pg_yr_mc_hi",
        "median_progress_pct", "q25_progress_pct", "q75_progress_pct",
    ]]
    sheet_d = curves[[
        "sensitivity_label", "saturation_gap_multiplier", "progress_pct", "mean_cram_pct",
        "cram_flux_pg_yr", "cram_flux_pg_yr_mc_lo", "cram_flux_pg_yr_mc_hi",
        "median_progress_pct", "q25_progress_pct", "q75_progress_pct",
    ]]
    with pd.ExcelWriter(XLSX, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
        sheet_c.to_excel(w, sheet_name="c", index=False)
        sheet_d.to_excel(w, sheet_name="d", index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-mc", type=int, default=N_MC)
    p.add_argument("--sat-tif", type=Path, default=None)
    args = p.parse_args()
    df = prepare(sat_tif(args.sat_tif))
    curves, sites = monte_carlo(df, args.n_mc)
    curves = smooth(curves)
    base = sites[sites["sensitivity_label"] == "1.0x original saturation"]
    progress = {
        "median_progress_pct": float(base["current_progress_pct"].median()),
        "q25_progress_pct": float(base["current_progress_pct"].quantile(0.25)),
        "q75_progress_pct": float(base["current_progress_pct"].quantile(0.75)),
    }
    write_cd(curves, progress)
    print("saved", XLSX, "sheets c, d")


if __name__ == "__main__":
    main()
