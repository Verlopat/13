import argparse
import json
import math
import os
from dataclasses import dataclass
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def lag_series(s, lag):
    return s.shift(lag)


def aligned_arrays(*series, dropna=True):
    df = pd.concat(series, axis=1)
    if dropna:
        df = df.dropna()
    return [df.iloc[:, i].to_numpy() for i in range(df.shape[1])]


def pearson_test(x, y):
    if len(x) < 3 or len(y) < 3:
        return 0.0, 1.0
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0, 1.0
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def partial_corr_test(x, y, z):
    if z is None or len(z) == 0:
        return pearson_test(x, y)
    z = np.asarray(z)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < len(z) + 3:
        return 0.0, 1.0
    try:
        x_res = sm.OLS(x, sm.add_constant(z, has_constant='add')).fit().resid
        y_res = sm.OLS(y, sm.add_constant(z, has_constant='add')).fit().resid
        return pearson_test(x_res, y_res)
    except Exception:
        return 0.0, 1.0


def gmm_estimate(y, x, z=None):
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)
    if z is None or len(z) == 0:
        X = sm.add_constant(x, has_constant='add')
        model = sm.OLS(y, X).fit()
        return float(model.params[1]), model
    z = np.asarray(z)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    Z = sm.add_constant(z, has_constant='add')
    try:
        fs = sm.OLS(x, Z).fit()
        x_hat = fs.fittedvalues.reshape(-1, 1)
        X2 = sm.add_constant(x_hat, has_constant='add')
        ss = sm.OLS(y, X2).fit()
        return float(ss.params[1]), ss
    except Exception:
        X = sm.add_constant(x, has_constant='add')
        model = sm.OLS(y, X).fit()
        return float(model.params[1]), model


@dataclass
class EdgeResult:
    source: str
    target: str
    lag: int
    score: float
    p_value: float
    coef: float
    method: str
    instruments: str
    proxies: str


def simulate_system(n=2000, seed=7, conf_strength=0.6, noise=0.5):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=n)
    v = rng.normal(size=n)
    z = rng.normal(size=n)
    x = np.zeros(n)
    y = np.zeros(n)
    w = np.zeros(n)
    c = np.zeros(n)
    for t in range(2, n):
        c[t] = 0.7 * c[t - 1] + 0.4 * v[t - 1] + rng.normal(scale=noise)
        x[t] = 0.6 * x[t - 1] + 0.3 * z[t - 1] + conf_strength * u[t - 1] + rng.normal(scale=noise)
        y[t] = 0.5 * y[t - 1] + 0.8 * x[t - 1] + conf_strength * u[t - 1] + 0.3 * c[t - 1] + rng.normal(scale=noise)
        w[t] = 0.4 * w[t - 1] + 0.9 * u[t - 1] + rng.normal(scale=noise)
    df = pd.DataFrame({"X": x, "Y": y, "W": w, "Z": z, "C": c})
    truth = [("X", "Y", 1), ("Z", "X", 1)]
    return df, truth


def candidate_lags(max_lag):
    return list(range(1, max_lag + 1))


def build_lagged_frame(df, target, source, lag, controls=None):
    y = df[target].iloc[lag:].reset_index(drop=True)
    x = df[source].shift(lag).iloc[lag:].reset_index(drop=True)
    cols = [x.rename(f"{source}_lag{lag}")]
    if controls:
        for c in controls:
            cols.append(df[c].shift(lag).iloc[lag:].reset_index(drop=True).rename(f"{c}_lag{lag}"))
    X = pd.concat(cols, axis=1)
    data = pd.concat([y.rename(target), X], axis=1).dropna()
    return data.iloc[:, 0].to_numpy(), data.iloc[:, 1:].to_numpy()


def screen_edges(df, cols, max_lag=2, alpha=0.05, min_abs_r=0.08):
    results = []
    for target in cols:
        for source in cols:
            if source == target:
                continue
            for lag in candidate_lags(max_lag):
                y = df[target].iloc[lag:].to_numpy()
                x = df[source].shift(lag).iloc[lag:].to_numpy()
                r, p = pearson_test(x, y)
                if np.isfinite(r) and abs(r) >= min_abs_r and p < alpha:
                    results.append((source, target, lag, r, p))
    return results


def choose_instruments(df, source, target, lag, cols, max_lag=2, alpha=0.05):
    instruments = []
    for z in cols:
        if z in (source, target):
            continue
        zx = df[z].shift(lag).iloc[lag:].to_numpy()
        x = df[source].shift(lag).iloc[lag:].to_numpy()
        y = df[target].iloc[lag:].to_numpy()
        r1, p1 = pearson_test(zx, x)
        r2, p2 = pearson_test(zx, y)
        if p1 < alpha and p2 > alpha:
            instruments.append(z)
    return instruments


def choose_proxies(df, target, source, lag, cols, alpha=0.05):
    proxies = []
    for p in cols:
        if p in (source, target):
            continue
        px = df[p].shift(lag).iloc[lag:].to_numpy()
        x = df[source].shift(lag).iloc[lag:].to_numpy()
        y = df[target].iloc[lag:].to_numpy()
        r1, p1 = pearson_test(px, x)
        r2, p2 = pearson_test(px, y)
        if p1 < alpha and p2 < alpha:
            proxies.append(p)
    return proxies


def fit_edge(df, source, target, lag, instruments, proxies):
    y = df[target].iloc[lag:].reset_index(drop=True)
    x = df[source].shift(lag).iloc[lag:].reset_index(drop=True).rename("x")
    frame = pd.DataFrame({"y": y, "x": x})
    inst_cols = []
    proxy_cols = []
    if instruments:
        for z in instruments:
            c = df[z].shift(lag).iloc[lag:].reset_index(drop=True).rename(f"z_{z}")
            frame[c.name] = c
            inst_cols.append(c.name)
    if proxies:
        for p in proxies:
            c = df[p].shift(lag).iloc[lag:].reset_index(drop=True).rename(f"p_{p}")
            frame[c.name] = c
            proxy_cols.append(c.name)
    frame = frame.dropna()
    yv = frame["y"].to_numpy()
    xv = frame["x"].to_numpy().reshape(-1, 1)
    if proxy_cols:
        Z = frame[proxy_cols].to_numpy()
        coef, model = gmm_estimate(yv, xv, Z)
        pval = float(model.pvalues[1]) if hasattr(model, "pvalues") and len(model.pvalues) > 1 else 1.0
        return coef, pval, "proxy_adjusted"
    if inst_cols:
        Z = frame[inst_cols].to_numpy()
        coef, model = gmm_estimate(yv, xv, Z)
        pval = float(model.pvalues[1]) if hasattr(model, "pvalues") and len(model.pvalues) > 1 else 1.0
        return coef, pval, "iv_2sls"
    X = sm.add_constant(xv, has_constant='add')
    model = sm.OLS(yv, X).fit()
    coef = float(model.params[1])
    pval = float(model.pvalues[1])
    return coef, pval, "ols"


def discover_graph(df, max_lag=2, alpha=0.05):
    cols = list(df.columns)
    screened = screen_edges(df, cols, max_lag=max_lag, alpha=alpha)
    edges = []
    G = nx.DiGraph()
    for c in cols:
        G.add_node(c)
    for source, target, lag, r, p in screened:
        inst = choose_instruments(df, source, target, lag, cols, max_lag=max_lag, alpha=alpha)
        prox = choose_proxies(df, target, source, lag, cols, alpha=alpha)
        coef, pval, method = fit_edge(df, source, target, lag, inst, prox)
        if np.isfinite(coef) and abs(coef) > 1e-8:
            edge = EdgeResult(
                source=source,
                target=target,
                lag=lag,
                score=float(abs(r)),
                p_value=float(pval),
                coef=float(coef),
                method=method,
                instruments=";".join(inst),
                proxies=";".join(prox),
            )
            edges.append(edge)
            G.add_edge(source, target, lag=lag, weight=float(abs(coef)), method=method)
    return G, edges


def truth_to_set(truth):
    return set(truth)


def pred_to_set(edges):
    return set((e.source, e.target, e.lag) for e in edges)


def evaluate_edges(edges, truth):
    pred = pred_to_set(edges)
    tru = truth_to_set(truth)
    tp = len(pred & tru)
    fp = len(pred - tru)
    fn = len(tru - pred)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    shd = fp + fn
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1, "shd": shd}


def save_outputs(outdir, df, G, edges, metrics):
    ensure_dir(outdir)
    df.to_csv(os.path.join(outdir, "synthetic_data.csv"), index=False)
    edf = pd.DataFrame([e.__dict__ for e in edges])
    edf.to_csv(os.path.join(outdir, "discovered_edges.csv"), index=False)
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    nx.write_edgelist(G, os.path.join(outdir, "graph.edgelist"), data=True)


def run(n=2000, seed=7, max_lag=2, alpha=0.05, conf_strength=0.6, noise=0.5, outdir="results"):
    df, truth = simulate_system(n=n, seed=seed, conf_strength=conf_strength, noise=noise)
    G, edges = discover_graph(df.drop(columns=[]), max_lag=max_lag, alpha=alpha)
    metrics = evaluate_edges(edges, truth)
    save_outputs(outdir, df, G, edges, metrics)
    return df, truth, G, edges, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_lag", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--conf_strength", type=float, default=0.6)
    parser.add_argument("--noise", type=float, default=0.5)
    parser.add_argument("--outdir", type=str, default="results")
    args = parser.parse_args()

    df, truth, G, edges, metrics = run(
        n=args.n,
        seed=args.seed,
        max_lag=args.max_lag,
        alpha=args.alpha,
        conf_strength=args.conf_strength,
        noise=args.noise,
        outdir=args.outdir,
    )

    print("Truth:", truth)
    print("Metrics:", json.dumps(metrics, indent=2))
    print("Edges:")
    for e in edges:
        print(
            {
                "source": e.source,
                "target": e.target,
                "lag": e.lag,
                "coef": round(e.coef, 4),
                "p_value": round(e.p_value, 6),
                "method": e.method,
                "instruments": e.instruments,
                "proxies": e.proxies,
            }
        )


if __name__ == "__main__":
    main()
