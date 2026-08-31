#!/usr/bin/env python3
"""PLS-SEM for Fig. 2c. Writes sheets c and c_r2 in fig2_data/fig2.xlsx.

Weathering        Mode A  DIC, Ca, Mg, Si
Hydrochemistry    Mode A  WT, DO, pH, TN, TP
Microbial_factors Mode A  ACE, Shannon, aerobic_chemoheterotrophy, phototrophy
DOM_content       Mode A  DOC, Auto
DOM_stability     Mode A  Auto-S, Allo-S, CRAM

  Weathering -> Hydrochemistry, Microbial_factors, DOM_stability
  Hydrochemistry -> Microbial_factors, DOM_content
  Microbial_factors -> DOM_content, DOM_stability
  DOM_content -> DOM_stability

python fig2/model_c.py
python fig2/model_c.py --n-boot 1000
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
XLSX = ROOT / "fig2_data" / "fig2.xlsx"

LATENTS = [
    "Weathering",
    "Hydrochemistry",
    "Microbial_factors",
    "DOM_content",
    "DOM_stability",
]
BLOCKS = {
    "Weathering": ["DIC", "Ca", "Mg", "Si"],
    "Hydrochemistry": ["WT", "DO", "pH", "TN", "TP"],
    "Microbial_factors": ["ACE", "Shannon", "aerobic_chemoheterotrophy", "phototrophy"],
    "DOM_content": ["DOC", "Auto"],
    "DOM_stability": ["Auto-S", "Allo-S", "CRAM"],
}
# PATH[to, from] = 1  means from -> to
PATH = np.array(
    [
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0],
    ],
    dtype=float,
)


def zscore(a):
    a = np.asarray(a, float)
    if a.ndim == 1:
        s = a.std()
        return (a - a.mean()) / s if s else a * 0.0
    s = a.std(axis=0, keepdims=True)
    s = np.where(s == 0, 1.0, s)
    return (a - a.mean(axis=0, keepdims=True)) / s


def load():
    df = pd.read_excel(XLSX, sheet_name="a", header=1)
    rename = {}
    for c in df.columns:
        s = str(c)
        if s == "CWS":
            rename[c] = "DIC"
        elif s != "DOC" and s.replace("\\", "").startswith("DO"):
            rename[c] = "DO"
    df = df.rename(columns=rename)
    cols = [v for vs in BLOCKS.values() for v in vs]
    out = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    return out.reset_index(drop=True)


def ols(X, y):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if X.ndim == 1:
        X = X[:, None]
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def fit(df):
    idx = {name: i for i, name in enumerate(LATENTS)}
    blocks = {k: [df.columns.get_loc(v) for v in vs] for k, vs in BLOCKS.items()}
    X = zscore(df.to_numpy(float))
    n, J = X.shape[0], len(LATENTS)
    Y = np.zeros((n, J))
    for name, cols in blocks.items():
        Y[:, idx[name]] = zscore(X[:, cols].mean(axis=1))

    for _ in range(300):
        Z = np.zeros_like(Y)
        for j, name in enumerate(LATENTS):
            z = np.zeros(n)
            preds = np.where(PATH[j] == 1)[0]
            succs = np.where(PATH[:, j] == 1)[0]
            if len(preds):
                z = z + Y[:, preds] @ ols(Y[:, preds], Y[:, j])
            for k in succs:
                r = np.corrcoef(Y[:, j], Y[:, k])[0, 1]
                if np.isfinite(r):
                    z = z + r * Y[:, k]
            if np.allclose(z, 0):
                z = Y[:, j]
            Z[:, j] = zscore(z)

        Ynew = np.zeros_like(Y)
        for name, cols in blocks.items():
            j = idx[name]
            w = X[:, cols].T @ Z[:, j] / n
            nrm = np.linalg.norm(w)
            if nrm == 0:
                w = np.ones(len(cols)) / len(cols)
            else:
                w = w / nrm
            y = zscore(X[:, cols] @ w)
            if np.corrcoef(y, X[:, cols[0]])[0, 1] < 0:
                y = -y
            Ynew[:, j] = y
        if np.max(np.abs(Ynew - Y)) < 1e-7:
            Y = Ynew
            break
        Y = Ynew

    P = np.zeros((J, J))
    r2 = np.zeros(J)
    types = []
    for j, name in enumerate(LATENTS):
        preds = np.where(PATH[j] == 1)[0]
        if len(preds) == 0:
            types.append("Exogenous")
            continue
        types.append("Endogenous")
        beta = ols(Y[:, preds], Y[:, j])
        P[j, preds] = beta
        resid = Y[:, j] - Y[:, preds] @ beta
        r2[j] = 1.0 - resid.var() / Y[:, j].var() if Y[:, j].var() else 0.0

    total = np.linalg.inv(np.eye(J) - P) - np.eye(J)
    indirect = total - P

    rows = []
    for fr, a in enumerate(LATENTS):
        for to, b in enumerate(LATENTS):
            if abs(total[to, fr]) < 1e-12:
                continue
            rows.append(
                {
                    "relationship": f"{a} -> {b}",
                    "direct": float(P[to, fr]),
                    "indirect": float(indirect[to, fr]),
                    "total": float(total[to, fr]),
                }
            )
    effects = pd.DataFrame(rows)
    inner = pd.DataFrame(
        {
            "Node": LATENTS,
            "Type": types,
            "R2": r2,
        }
    )
    return effects, inner


def p_stars(p):
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.1:
        return "**"
    if p < 0.5:
        return "*"
    return ""


def choose(direct, indirect, dp, ip):
    d_ok = np.isfinite(direct) and abs(direct) > 0
    i_ok = np.isfinite(indirect) and abs(indirect) > 0
    d_sig = d_ok and np.isfinite(dp)
    i_sig = i_ok and np.isfinite(ip)
    if d_sig and i_sig:
        return ("indirect", indirect, ip) if ip < dp else ("direct", direct, dp)
    if d_sig:
        return "direct", direct, dp
    if i_sig:
        return "indirect", indirect, ip
    if d_ok:
        return "direct", direct, np.nan
    if i_ok:
        return "indirect", indirect, np.nan
    return None, np.nan, np.nan


def bootstrap(df, effects, n_boot, seed=123):
    rng = np.random.default_rng(seed)
    rels = effects["relationship"].tolist()
    dmat = np.full((n_boot, len(rels)), np.nan)
    imat = np.full((n_boot, len(rels)), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, len(df), len(df))
        try:
            boot, _ = fit(df.iloc[idx].reset_index(drop=True))
        except np.linalg.LinAlgError:
            continue
        m = boot.set_index("relationship")
        for i, rel in enumerate(rels):
            if rel in m.index:
                dmat[b, i] = m.loc[rel, "direct"]
                imat[b, i] = m.loc[rel, "indirect"]

    def se_p(orig, mat):
        se = np.nanstd(mat, axis=0, ddof=1)
        p = np.array(
            [
                2 * norm.sf(abs(o / s)) if np.isfinite(o) and np.isfinite(s) and s > 0 else np.nan
                for o, s in zip(orig, se)
            ]
        )
        return p

    out = effects.copy()
    out["direct_p_value"] = se_p(out["direct"].to_numpy(), dmat)
    out["direct_stars"] = [p_stars(p) for p in out["direct_p_value"]]
    out["indirect_p_value"] = se_p(out["indirect"].to_numpy(), imat)
    out["indirect_stars"] = [p_stars(p) for p in out["indirect_p_value"]]
    chosen = [
        choose(d, i, dp, ip)
        for d, i, dp, ip in zip(
            out["direct"], out["indirect"], out["direct_p_value"], out["indirect_p_value"]
        )
    ]
    out["chosen_type"] = [c[0] for c in chosen]
    out["chosen_est"] = [c[1] for c in chosen]
    out["chosen_p"] = [c[2] for c in chosen]
    out["chosen_stars"] = [p_stars(p) for p in out["chosen_p"]]
    return out


def save_sheets(effects, inner):
    with pd.ExcelWriter(XLSX, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
        effects.to_excel(w, sheet_name="c", index=False)
        inner.to_excel(w, sheet_name="c_r2", index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-boot", type=int, default=1000)
    args = p.parse_args()
    df = load()
    effects, inner = fit(df)
    effects = bootstrap(df, effects, args.n_boot)
    save_sheets(effects, inner)
    print("n =", len(df), "bootstrap =", args.n_boot)
    print(effects[["relationship", "chosen_type", "chosen_est", "chosen_stars"]].to_string(index=False))
    print(inner.to_string(index=False))
    print("saved", XLSX, "sheets c, c_r2")


if __name__ == "__main__":
    main()
