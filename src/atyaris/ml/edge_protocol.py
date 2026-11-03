"""Fixed evaluation protocol for proving a model edge over the closing market.

Universe: races whose every runner has closing odds (``market_complete == 1``).
Training pool: races with >=80% odds coverage (missing runners imputed).
Final holdout: last 20% of evaluable days, used once via ``run_protocol(final=True)``.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from atyaris.ml.backtest import NEW_LONGSHOT_FEATURES
from atyaris.ml.features import TJK_SELECTED_STAGE1_FEATURE_COLUMNS

PROTOCOL_VERSION = "edge-protocol-v1"
LONGSHOT_ODDS = 10.0
RETRAIN_DAYS = 14
WARMUP_DAYS = 45
HOLDOUT_FRACTION = 0.20
BOOTSTRAP_ITERATIONS = 1000
POOL_COVERAGE = 0.80
ALPHAS = (1.0, 10.0, 100.0, 1000.0)
TRAINER_FEATURES = ("trainer_win_rate", "trainer_top3_rate", "trainer_top5_rate", "trainer_experience_log")
_EXCLUDED = {"distance", "field_size", "field_size_local"}


def base_features(frame: pd.DataFrame) -> list[str]:
    cols = []
    for c in TJK_SELECTED_STAGE1_FEATURE_COLUMNS:
        low = c.lower()
        if c in _EXCLUDED or "odds" in low or "market" in low or c in NEW_LONGSHOT_FEATURES:
            continue
        if c in frame.columns and pd.api.types.is_numeric_dtype(frame[c]):
            cols.append(c)
    return cols


def prepare_frame(features_csv: Path, cache: Path | None = None) -> pd.DataFrame:
    if cache is not None and cache.exists() and cache.stat().st_mtime >= features_csv.stat().st_mtime:
        return pickle.loads(cache.read_bytes())
    df = pd.read_csv(features_csv, low_memory=False)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    valid = df["odds"].fillna(0) > 1.0
    df["has_odds"] = valid
    inv = np.where(valid, 1.0 / df["odds"].where(valid, 1.0), 0.0)
    df["_inv"] = inv
    g = df.groupby("race_id")
    df["coverage"] = g["has_odds"].transform("mean")
    df["n_runners"] = g["race_id"].transform("size")
    df["n_winners"] = g["is_winner"].transform("sum")
    strict = df["coverage"] == 1.0
    overround = float(df.loc[strict].groupby("race_id")["_inv"].sum().median()) if strict.any() else 1.17
    known = g["_inv"].transform("sum")
    n_missing = (~df["has_odds"]).groupby(df["race_id"]).transform("sum")
    fill = np.where(n_missing > 0, np.maximum(overround - known, 0.0) / n_missing.clip(lower=1), 0.0)
    raw = np.where(df["has_odds"], df["_inv"], fill)
    df["_raw"] = raw
    df["q"] = raw / df.groupby("race_id")["_raw"].transform("sum")
    df["strict"] = strict
    df["in_pool"] = (df["coverage"] >= POOL_COVERAGE) & (df["n_winners"] == 1) & (df["n_runners"] >= 4)
    df = df[df["in_pool"]].sort_values(["date", "race_id"]).reset_index(drop=True)
    df["is_winner"] = df["is_winner"].astype(float)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps(df))
    return df


def split_days(df: pd.DataFrame) -> tuple[list, list]:
    days = sorted(df.loc[df["strict"], "date"].unique())
    cut = int(round(len(days) * (1 - HOLDOUT_FRACTION)))
    return days[:cut], days[cut:]


# ---------------------------------------------------------------- models
def _starts(race_ids: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.r_[True, race_ids[1:] != race_ids[:-1]])


def _softmax_groups(u: np.ndarray, starts: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    m = np.repeat(np.maximum.reduceat(u, starts), sizes)
    e = np.exp(u - m)
    return e / np.repeat(np.add.reduceat(e, starts), sizes)


def fit_cl(X, offset, y, race_ids, alpha):
    starts = _starts(race_ids)
    sizes = np.diff(np.r_[starts, len(race_ids)])

    def fun(w):
        p = _softmax_groups(offset + X @ w, starts, sizes)
        ll = -np.log(np.clip(p[y == 1], 1e-12, None)).sum() + 0.5 * alpha * (w @ w)
        return ll, X.T @ (p - y) + alpha * w

    return minimize(fun, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 200}).x


def _design(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]):
    mu = train[cols].mean()
    sd = train[cols].std().replace(0, 1).fillna(1)
    f = lambda d: ((d[cols] - mu) / sd).fillna(0.0).clip(-5, 5).to_numpy(float)
    return f(train), f(test)


def _logq(d):
    return np.log(np.clip(d["q"].to_numpy(float), 1e-6, None))


def _renorm(d, p):
    return pd.Series(p, index=d.index).groupby(d["race_id"]).transform(lambda s: s / s.sum()).to_numpy()


def _select_alpha(train, cols, free_offset):
    days = sorted(train["date"].unique())
    cut = days[int(len(days) * 0.75)]
    a, b = train[train["date"] < cut], train[train["date"] >= cut]
    best, best_ll = ALPHAS[-1], np.inf
    for al in ALPHAS:
        p = _cl_predict(a, b, cols, al, free_offset)
        ll = -np.log(np.clip(p[b["is_winner"].to_numpy() == 1], 1e-12, None)).mean()
        if ll < best_ll:
            best, best_ll = al, ll
    return best


def _cl_predict(train, test, cols, alpha, free_offset):
    Xa, Xb = _design(train, test, cols)
    if free_offset:
        Xa = np.c_[Xa, _logq(train)]
        Xb = np.c_[Xb, _logq(test)]
        off_a = off_b = 0.0
        w = fit_cl(Xa, np.zeros(len(Xa)), train["is_winner"].to_numpy(), train["race_id"].to_numpy(), alpha)
        return _renorm(test, np.exp(Xb @ w))
    w = fit_cl(Xa, _logq(train), train["is_winner"].to_numpy(), train["race_id"].to_numpy(), alpha)
    return _renorm(test, np.exp(_logq(test) + Xb @ w))


def make_cl(cols_fn: Callable[[pd.DataFrame], list[str]], free_offset=False):
    def predict(train, test):
        cols = cols_fn(train)
        alpha = _select_alpha(train, cols, free_offset)
        return _cl_predict(train, test, cols, alpha, free_offset)

    return predict


def make_gbm(cols_fn):
    from sklearn.ensemble import HistGradientBoostingClassifier

    def fit_pred(a, b, cols, wts):
        m = HistGradientBoostingClassifier(
            max_depth=3, max_iter=120, learning_rate=0.05, l2_regularization=5.0,
            min_samples_leaf=200, random_state=0,
        )
        m.fit(np.c_[a[cols].to_numpy(float), _logq(a)], a["is_winner"].to_numpy())
        raw = m.predict_proba(np.c_[b[cols].to_numpy(float), _logq(b)])[:, 1]
        pg = _renorm(b, raw)
        return pg

    def predict(train, test):
        cols = cols_fn(train)
        days = sorted(train["date"].unique())
        cut = days[int(len(days) * 0.75)]
        a, b = train[train["date"] < cut], train[train["date"] >= cut]
        pg = fit_pred(a, b, cols, None)
        best_w, best_ll = 0.0, np.inf
        for w in (0.0, 0.25, 0.5, 1.0):
            p = _renorm(b, np.exp((1 - w) * _logq(b) + w * np.log(np.clip(pg, 1e-9, None))))
            ll = -np.log(np.clip(p[b["is_winner"].to_numpy() == 1], 1e-12, None)).mean()
            if ll < best_ll:
                best_w, best_ll = w, ll
        pt = fit_pred(train, test, cols, None)
        return _renorm(test, np.exp((1 - best_w) * _logq(test) + best_w * np.log(np.clip(pt, 1e-9, None))))

    return predict


# ------------------------------------------------------------ walk-forward
def walk_forward(df: pd.DataFrame, predictor, test_days: list, strict_only: bool = True) -> pd.DataFrame:
    """Predict strict test races block by block; training uses pool races strictly before each block."""
    out = []
    first = df["date"].min()
    i = 0
    while i < len(test_days):
        start = test_days[i]
        end = start + pd.Timedelta(days=RETRAIN_DAYS)
        block_days = [d for d in test_days if start <= d < end]
        i += len(block_days)
        train = df[df["date"] < start]
        if (start - first).days < WARMUP_DAYS or train["race_id"].nunique() < 150:
            continue
        test = df[df["date"].isin(block_days) & (df["strict"] if strict_only else True)].copy()
        if test.empty:
            continue
        test["p"] = predictor(train, test)
        out.append(test)
    return pd.concat(out) if out else pd.DataFrame()


# ----------------------------------------------------------------- metrics
def _race_table(pred: pd.DataFrame, tau: float | None) -> pd.DataFrame:
    win = pred[pred["is_winner"] == 1].set_index("race_id")
    t = pd.DataFrame({"date": win["date"], "ll_m": -np.log(win["q"].clip(lower=1e-9)),
                      "ll_p": -np.log(win["p"].clip(lower=1e-9))})
    return t


def _bets(pred: pd.DataFrame, tau: float) -> pd.DataFrame:
    ev = pred["p"] * pred["odds"] - 1.0
    b = pred[ev > tau].copy()
    b["profit"] = np.where(b["is_winner"] == 1, b["odds"] - 1.0, -1.0)
    kelly = (0.25 * ev[b.index] / (b["odds"] - 1.0)).clip(0, 0.05)
    b["k_stake"] = kelly
    b["k_profit"] = b["k_stake"] * b["profit"]
    return b


def choose_tau(pred: pd.DataFrame, grid=(0.0, 0.1, 0.2, 0.3, 0.5), default=0.1, min_bets=30) -> float:
    best, best_roi = default, -np.inf
    for t in grid:
        b = _bets(pred, t)
        if len(b) >= min_bets and b["profit"].mean() > best_roi:
            best, best_roi = t, b["profit"].mean()
    return best


def _cluster_ci(day_stats: pd.DataFrame, num: str, den: str | None, alpha: float, rng) -> list[float]:
    arr_n = day_stats[num].to_numpy(float)
    arr_d = day_stats[den].to_numpy(float) if den else np.ones(len(day_stats))
    n = len(arr_n)
    idx = rng.integers(0, n, size=(BOOTSTRAP_ITERATIONS, n))
    d = arr_d[idx].sum(1)
    v = np.where(d > 0, arr_n[idx].sum(1) / np.where(d > 0, d, 1), np.nan)
    return [float(np.nanquantile(v, alpha / 2)), float(np.nanquantile(v, 1 - alpha / 2))]


def evaluate(pred: pd.DataFrame, tau: float = 0.1, n_comparisons: int = 6, seed: int = 7) -> dict:
    if pred.empty:
        return {"n_races": 0}
    rng = np.random.default_rng(seed)
    t = _race_table(pred, tau)
    t["d"] = t["ll_p"] - t["ll_m"]
    day = t.groupby("date").agg(d=("d", "sum"), n=("d", "size"), llp=("ll_p", "sum"), llm=("ll_m", "sum"))
    res = {
        "n_races": int(len(t)), "n_days": int(day.shape[0]),
        "logloss_model": float(t["ll_p"].mean()), "logloss_market": float(t["ll_m"].mean()),
        "delta_logloss": float(t["d"].mean()),  # negative = model better
        "delta_ci95": _cluster_ci(day, "d", "n", 0.05, rng),
        "delta_ci_bonferroni": _cluster_ci(day, "d", "n", 0.05 / n_comparisons, rng),
    }
    # longshot winners only (market already says <=10% win chance)
    ls = pred[(pred["odds"] >= LONGSHOT_ODDS)]
    res["longshot"] = {
        "n_runners": int(len(ls)), "n_winners": int(ls["is_winner"].sum()),
        "market_mean_p": float(ls["q"].mean()) if len(ls) else None,
        "model_mean_p": float(ls["p"].mean()) if len(ls) else None,
        "actual_rate": float(ls["is_winner"].mean()) if len(ls) else None,
    }
    buckets = pd.cut(pred["odds"], [0, 4, 10, 25, 1e9], labels=["<4", "4-10", "10-25", "25+"], right=False)
    res["calibration"] = {
        str(k): {"n": int(len(g)), "actual": float(g["is_winner"].mean()), "model": float(g["p"].mean()),
                 "market": float(g["q"].mean())}
        for k, g in pred.groupby(buckets, observed=True)
    }
    b = _bets(pred, tau)
    res["tau"] = tau
    for name, sub in (("all", b), ("longshot", b[b["odds"] >= LONGSHOT_ODDS])):
        if sub.empty:
            res[f"flat_{name}"] = {"n_bets": 0}
            continue
        ds = sub.groupby("date").agg(pr=("profit", "sum"), n=("profit", "size"), kp=("k_profit", "sum"),
                                     ks=("k_stake", "sum"))
        res[f"flat_{name}"] = {
            "n_bets": int(len(sub)), "roi": float(sub["profit"].mean()),
            "roi_ci95": _cluster_ci(ds, "pr", "n", 0.05, rng),
            "roi_ci_bonferroni": _cluster_ci(ds, "pr", "n", 0.05 / n_comparisons, rng),
            "kelly_roi": float(sub["k_profit"].sum() / max(sub["k_stake"].sum(), 1e-9)),
        }
    # market baseline flat bet on the same selections (takeout cost reference)
    res["market_roi_same_selections"] = float(b["profit"].mean() * 0 + (b["q"] * b["odds"] - 1).mean()) if len(b) else None
    top = pred.assign(rk=pred.groupby("race_id")["p"].rank(ascending=False, method="first"))
    top4 = top[top["rk"] <= 4]
    hit = (top4["finish_position"].fillna(99) <= 4)
    res["top4_precision"] = {
        "all": float(hit.mean()),
        "favorite(<10)": float(hit[top4["odds"] < LONGSHOT_ODDS].mean()) if (top4["odds"] < LONGSHOT_ODDS).any() else None,
        "longshot(>=10)": float(hit[top4["odds"] >= LONGSHOT_ODDS].mean()) if (top4["odds"] >= LONGSHOT_ODDS).any() else None,
    }
    return res


def market_prediction(test_df: pd.DataFrame) -> pd.DataFrame:
    out = test_df[test_df["strict"]].copy()
    out["p"] = out["q"]
    return out


# ------------------------------------------------------------------ driver
def candidates(df: pd.DataFrame) -> dict[str, Callable]:
    base = base_features(df)
    longshot = [c for c in NEW_LONGSHOT_FEATURES if c in df.columns]
    trainer = [c for c in TRAINER_FEATURES if c in df.columns]
    no_w = [c for c in base if "weight" not in c]
    return {
        "C0_free_logq_cl": make_cl(lambda d: base, free_offset=True),
        "C1_offset_cl": make_cl(lambda d: base),
        "C1_no_weight": make_cl(lambda d: no_w),
        "C1_longshot5": make_cl(lambda d: base + longshot),
        "C1_trainer": make_cl(lambda d: base + trainer),
        "C2_gbm": make_gbm(lambda d: base),
    }


def run_protocol(features_csv: Path, cache_dir: Path, final: bool = False, names: list[str] | None = None) -> dict:
    df = prepare_frame(features_csv, cache_dir / "edge_frame.pkl")
    dev_days, hold_days = split_days(df)
    days = hold_days if final else dev_days
    results: dict = {"protocol": PROTOCOL_VERSION, "stage": "holdout" if final else "dev",
                     "dev_days": len(dev_days), "holdout_days": len(hold_days),
                     "pool_races": int(df["race_id"].nunique()),
                     "strict_races": int(df.loc[df["strict"], "race_id"].nunique()), "models": {}}
    cands = candidates(df)
    for name in names or list(cands):
        cp = cache_dir / f"edge_pred_{results['stage']}_{name}.pkl"
        if cp.exists() and cp.stat().st_mtime > cache_dir.joinpath("edge_frame.pkl").stat().st_mtime:
            pred = pickle.loads(cp.read_bytes())
        else:
            pred = walk_forward(df, cands[name], days)
            cp.write_bytes(pickle.dumps(pred))
        half = len(pred) // 2
        tau = choose_tau(pred.iloc[:half]) if len(pred) else 0.1
        results["models"][name] = evaluate(pred, tau)
    mp = walk_forward(df, lambda tr, te: te["q"].to_numpy(), days)
    results["models"]["M0_market"] = evaluate(mp, 0.1)
    return results


def save(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")


# ------------------------------------------------------------- v2 (yeni veri)
V2_CANDIDATES = ("C1_offset_cl", "C1_longshot5", "C1_trainer", "C2_gbm")


def holdout_registry(df: pd.DataFrame, used_days: list[str], path: Path) -> dict:
    """Eski kilitli holdout gunleri 'kullanilmis' sayilir; temiz holdout = kalan strict gunlerin son %20'si."""
    used = {pd.Timestamp(d) for d in used_days}
    days = sorted(df.loc[df["strict"], "date"].unique())
    fresh = [d for d in days if d not in used]
    cut = int(round(len(fresh) * (1 - HOLDOUT_FRACTION)))
    reg = {
        "used_old_holdout_days": sorted(str(d.date()) for d in used),
        "overlap_with_new_holdout": sorted(str(d.date()) for d in fresh[cut:] if d in used),
        "naive_last20_overlap_with_used": sum(1 for d in days[int(round(len(days) * 0.8)):] if d in used),
        "naive_last20_days": len(days) - int(round(len(days) * 0.8)),
        "dev_days": [str(d.date()) for d in fresh[:cut]],
        "clean_holdout_days": [str(d.date()) for d in fresh[cut:]],
    }
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reg, indent=1), encoding="utf-8")
    return reg


def run_v2(features_csv: Path, cache_dir: Path, registry_path: Path, used_days: list[str]) -> dict:
    df = prepare_frame(features_csv, cache_dir / "edge_frame.pkl")
    reg = holdout_registry(df, used_days, registry_path)
    dev = [pd.Timestamp(d) for d in reg["dev_days"]]
    hold = [pd.Timestamp(d) for d in reg["clean_holdout_days"]]
    cands = candidates(df)
    frame_m = (cache_dir / "edge_frame.pkl").stat().st_mtime

    def predict(name: str, days: list, tag: str) -> pd.DataFrame:
        cp = cache_dir / f"edge_pred_{tag}_{name}.pkl"
        if cp.exists() and cp.stat().st_mtime > frame_m:
            return pickle.loads(cp.read_bytes())
        pred = walk_forward(df, cands[name], days)
        cp.write_bytes(pickle.dumps(pred))
        return pred

    out: dict = {"protocol": "edge-protocol-v2", "strict_races": int(df.loc[df["strict"], "race_id"].nunique()),
                 "dev_days": len(dev), "holdout_days": len(hold),
                 "overlap_with_old_holdout": len(reg["overlap_with_new_holdout"]), "dev": {}}
    for name in V2_CANDIDATES:
        pred = predict(name, dev, "dev2")
        half = len(pred) // 2
        out["dev"][name] = evaluate(pred, choose_tau(pred.iloc[:half]) if len(pred) else 0.1, n_comparisons=4)
    best = min(out["dev"], key=lambda n: out["dev"][n].get("delta_logloss", 9))
    pred = predict(best, hold, "hold2")
    tau = choose_tau(predict(best, dev, "dev2"))
    h = evaluate(pred, tau, n_comparisons=1)
    out["selected"], out["holdout"] = best, h
    ci = h.get("delta_ci95", [0, 0])
    out["decision"] = "PROMOTE" if (ci[1] < 0 and h.get("flat_all", {}).get("roi", -1) > 0) else "DO_NOT_PROMOTE"
    out["status"] = "established" if out["decision"] == "PROMOTE" else "not_established"
    return out
