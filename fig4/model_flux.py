#!/usr/bin/env python3
"""ExtraTrees river flux for Fig. 4 a–b. Writes fig4.xlsx sheet b."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree
from shapely.geometry import Point
from sklearn.ensemble import ExtraTreesRegressor

HERE = Path(__file__).resolve().parent
XLSX = HERE / "fig4.xlsx"
IONS = Path("/home/pangyu/aoe/ions")
INPUT = IONS / "cram-final" / "input_data"

OBSERVED_XLSX = INPUT / "final_ions_data.xlsx"
DOC_XLSX = INPUT / "Fig2_source_data.xlsx"
ALK_PH_CSV = IONS / "worldwide_estimated_pH_and_alkalinity_at_HydroBasins_HYBASID.csv"
TRAIN_TEMP_CSV = INPUT / "sample_temp_nasa_power_2016_2025_0522.csv"
DOC_TEMP_TIF = INPUT / "global_temp_era5_2m_mean_2016_2025_0526.tif"
SAT_TIF = IONS / "2026-4-22" / "DIC_mmol_L.tif"
CARBONATE_SHP = IONS / "2026-4-22" / "Carbonate_rock_distribution.shp"
DISCHARGE_CSV = IONS / "cram-final" / "data" / "doc_points_cram_discharge_flux_0526.csv"

DIC_C = 12.01
ALK_UEQ = 1000.0
SEC_YR = 31.536


def xyz(lon, lat):
    lon = np.deg2rad(np.asarray(lon, float))
    lat = np.deg2rad(np.asarray(lat, float))
    c = np.cos(lat)
    return np.column_stack((c * np.cos(lon), c * np.sin(lon), np.sin(lat)))


class BasinLookup:
    def __init__(self, path):
        df = pd.read_csv(path, usecols=[
            "HYBAS_ID", "basin_longitude", "basin_latitude",
            "estimated_alkalinity", "estimated_pH",
        ])
        for c in ("basin_longitude", "basin_latitude", "estimated_alkalinity", "estimated_pH"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["basin_longitude", "basin_latitude", "estimated_pH"])
        self.alk = df["estimated_alkalinity"].to_numpy(float)
        self.ph = df["estimated_pH"].to_numpy(float)
        self.tree = cKDTree(xyz(df["basin_longitude"], df["basin_latitude"]))

    def attach(self, df):
        _, idx = self.tree.query(xyz(df["lon"], df["lat"]), k=1)
        out = df.copy()
        out["estimated_alkalinity"] = self.alk[idx]
        out["estimated_pH"] = self.ph[idx]
        return out


def zscore(a, b):
    m, s = float(np.nanmean(a)), float(np.nanstd(a))
    if not np.isfinite(s) or s == 0:
        s = 1.0
    return (a - m) / s, (b - m) / s, m, s


def design(train, test):
    td, xd, dm, ds = zscore(
        np.log1p(np.clip(train["DIC_obs_mmol_L"].to_numpy(float), 0, None)),
        np.log1p(np.clip(test["DIC_obs_mmol_L"].to_numpy(float), 0, None)),
    )
    tc, xc, cm, cs = zscore(
        np.log1p(np.clip(train["Ca_obs_mmol_L"].to_numpy(float), 0, None)),
        np.log1p(np.clip(test["Ca_obs_mmol_L"].to_numpy(float), 0, None)),
    )
    tp, xp, pm, ps = zscore(
        train["estimated_pH"].to_numpy(float), test["estimated_pH"].to_numpy(float),
    )
    stats = dict(log_dic_mean=dm, log_dic_sd=ds, log_ca_mean=cm, log_ca_sd=cs, ph_mean=pm, ph_sd=ps)
    Xtr = np.column_stack([td, tc, train["is_carbonate"].astype(int), train["temp_c"].to_numpy(float), tp])
    Xte = np.column_stack([xd, xc, test["is_carbonate"].astype(int), test["temp_c"].to_numpy(float), xp])
    return Xtr, Xte, stats


def design_apply(dic, ca, carb, temp, ph, stats):
    return np.column_stack([
        (np.log1p(np.clip(dic, 0, None)) - stats["log_dic_mean"]) / stats["log_dic_sd"],
        (np.log1p(np.clip(ca, 0, None)) - stats["log_ca_mean"]) / stats["log_ca_sd"],
        np.asarray(carb, int),
        np.asarray(temp, float),
        (np.asarray(ph, float) - stats["ph_mean"]) / stats["ph_sd"],
    ])


def carbonate_flag(df):
    pts = gpd.GeoDataFrame(
        df.copy(),
        geometry=[Point(float(x), float(y)) for x, y in zip(df["lon"], df["lat"])],
        crs="EPSG:4326",
    )
    carb = gpd.read_file(CARBONATE_SHP)
    if carb.crs is None:
        carb = carb.set_crs("EPSG:4326")
    carb = carb[carb["rock_type"].isin({1, 2})].to_crs("EPSG:4326")
    hits = set(gpd.sjoin(pts, carb[["geometry"]], how="inner", predicate="within").index.unique())
    out = df.copy()
    out["is_carbonate"] = out.index.map(lambda i: 1 if i in hits else 0).astype(int)
    return out


def sample_tif(path, lon, lat):
    out = np.full(len(lon), np.nan)
    with rasterio.open(path) as src:
        nodata, bounds, wrap = src.nodata, src.bounds, src.bounds.right > 180
        for i, (x, y) in enumerate(zip(lon, lat)):
            if pd.isna(x) or pd.isna(y):
                continue
            x, y = float(x), float(y)
            if wrap and x < 0:
                x += 360.0
            if not (bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top):
                continue
            v = next(src.sample([(x, y)]))[0]
            if nodata is not None and v == nodata:
                continue
            if np.isfinite(v):
                out[i] = float(v)
    return out


def load_train(lookup):
    df = pd.read_excel(OBSERVED_XLSX).rename(columns={
        "Sites": "sample", "CRAM(%)": "CRAM_obs_pct",
        "DIC(mg/L)": "DIC_obs_mg_L", "Ca2+(mg/L)": "Ca_obs_mg_L",
    })
    for c in ("lon", "lat", "CRAM_obs_pct", "DIC_obs_mg_L", "Ca_obs_mg_L"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["sample", "lon", "lat", "CRAM_obs_pct", "DIC_obs_mg_L", "Ca_obs_mg_L"])
    df["DIC_obs_mmol_L"] = df["DIC_obs_mg_L"].clip(lower=0) / DIC_C
    df["Ca_obs_mmol_L"] = df["Ca_obs_mg_L"].clip(lower=0) / 40.078
    df = carbonate_flag(df)
    temp = pd.read_csv(TRAIN_TEMP_CSV)
    temp["temp_c"] = pd.to_numeric(temp["temp_c"], errors="coerce")
    df = df.merge(temp[["sample", "temp_c"]], on="sample", how="left")
    df = lookup.attach(df)
    return df.dropna(subset=["temp_c", "estimated_pH"]).reset_index(drop=True)


def load_doc(lookup):
    df = pd.read_excel(DOC_XLSX, sheet_name="Predicted_DOC_conc").rename(columns={"DOC conc(mg/L)": "doc_mg_L"})
    for c in ("lon", "lat", "doc_mg_L"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["lon", "lat", "doc_mg_L"]).reset_index(drop=True)
    df = carbonate_flag(df)
    df = lookup.attach(df)
    df["dic_contemporary_mmol_L"] = df["estimated_alkalinity"] / ALK_UEQ
    df["temp_c"] = sample_tif(DOC_TEMP_TIF, df["lon"], df["lat"])
    return df


def predict_cram(df, model, stats):
    dic = pd.to_numeric(df["dic_contemporary_mmol_L"], errors="coerce").to_numpy(float)
    ca = dic / 2.0
    temp = pd.to_numeric(df["temp_c"], errors="coerce").to_numpy(float)
    ph = pd.to_numeric(df["estimated_pH"], errors="coerce").to_numpy(float)
    carb = pd.to_numeric(df["is_carbonate"], errors="coerce").to_numpy(float)
    ok = np.isfinite(dic) & (dic >= 0) & np.isfinite(ph) & np.isfinite(carb) & np.isfinite(temp)
    pred = np.full(len(df), np.nan)
    if ok.any():
        pred[ok] = model.predict(design_apply(dic[ok], ca[ok], carb[ok], temp[ok], ph[ok], stats))
    return pred


def main():
    need = [OBSERVED_XLSX, DOC_XLSX, ALK_PH_CSV, TRAIN_TEMP_CSV, DOC_TEMP_TIF, CARBONATE_SHP, DISCHARGE_CSV]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        raise FileNotFoundError("local inputs missing; plot with fig4.xlsx\n" + "\n".join(missing))

    lookup = BasinLookup(ALK_PH_CSV)
    train = load_train(lookup)
    X, _, stats = design(train, train)
    model = ExtraTreesRegressor(n_estimators=1400, min_samples_leaf=2, max_features=0.7, random_state=42, n_jobs=-1)
    model.fit(X, train["CRAM_obs_pct"].to_numpy())

    doc = load_doc(lookup)
    doc["cram"] = predict_cram(doc, model, stats)
    doc["doc_x_cram"] = doc["doc_mg_L"] * doc["cram"] / 100.0

    q = pd.read_csv(DISCHARGE_CSV, usecols=["lon", "lat", "doc_mg_L", "discharge_m3_s"])
    m = q.merge(doc[["lon", "lat", "doc_mg_L", "doc_x_cram"]], on=["lon", "lat", "doc_mg_L"])
    m["flux_contemporary_ton_yr"] = m["doc_x_cram"] * pd.to_numeric(m["discharge_m3_s"], errors="coerce") * SEC_YR
    out = m[["lon", "lat", "flux_contemporary_ton_yr"]].replace([np.inf, -np.inf], np.nan).dropna()
    out = out[(out["lat"] > -60) & (out["lat"] <= 85)]
    with pd.ExcelWriter(XLSX, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
        out.to_excel(w, sheet_name="b", index=False)
    print("saved", XLSX, "sheet b n=", len(out))


if __name__ == "__main__":
    main()
