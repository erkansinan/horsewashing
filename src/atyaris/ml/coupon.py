"""6'li ganyan kupon optimizasyonu: kalibre olasilik + butce altinda kapsama maksimizasyonu.

Degerlendirme gun-cluster bootstrap ile; kilitli son %20 altili gunu tek seferlik kullanilir.
Ikramiye ROI'si yayinlanan ikramiye biriminin 1 birim bahis olarak alinmasiyla hesaplanir.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from atyaris.ml import edge_protocol as E
from atyaris.ml.local_pipeline import _track_key

BUDGETS = (48, 144, 432)
BOOT = 1000


def load_altili(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if '"altili"' in line]
    a = pd.DataFrame(rows)
    a = a[a["winners"].map(len) == 6].copy()
    a["tk"] = a["hippodrome"].map(_track_key)
    return a.drop_duplicates(["race_date", "tk", "last_race_no", "ordinal"])


def calibrate_online(pred: pd.DataFrame, block_days: int = 14) -> pd.DataFrame:
    pred = pred.sort_values(["date", "race_id"]).copy()
    pred["p_cal"] = pred["p"]
    start = pred["date"].min()
    block = ((pred["date"] - start).dt.days // block_days).to_numpy()
    for b in np.unique(block):
        prior = pred[block < b]
        if len(prior) < 500:
            continue
        iso = IsotonicRegression(y_min=1e-4, y_max=1.0, out_of_bounds="clip").fit(prior["p"], prior["is_winner"])
        mask = block == b
        pred.loc[mask, "p_cal"] = np.clip(iso.predict(pred.loc[mask, "p"]), 1e-4, None)
    s = pred.groupby("race_id")["p_cal"].transform("sum")
    pred["p_cal"] = pred["p_cal"] / s
    return pred


def allocate(legs: list[np.ndarray], budget: int) -> list[int]:
    """Greedy: marjinal log-kapsama kazanci / marjinal maliyet en yuksek ayaga at ekle."""
    ks = [1] * len(legs)
    cs = [np.cumsum(np.sort(p)[::-1]) for p in legs]
    while True:
        cost = int(np.prod(ks))
        best, best_gain = None, 0.0
        for i, c in enumerate(cs):
            if ks[i] >= len(c):
                continue
            new_cost = cost // ks[i] * (ks[i] + 1)
            if new_cost > budget:
                continue
            gain = (np.log(c[ks[i]]) - np.log(c[ks[i] - 1])) / (new_cost - cost)
            if gain > best_gain:
                best, best_gain = i, gain
        if best is None:
            return ks
        ks[best] += 1


def fixed_alloc(legs: list[np.ndarray], budget: int) -> list[int]:
    k = max(1, int(budget ** (1 / len(legs)) + 1e-9))
    return [min(k, len(p)) for p in legs]


def build_days(pred: pd.DataFrame, altili: pd.DataFrame) -> list[dict]:
    pred = pred.copy()
    pred["tk"] = pred["track"].map(_track_key)
    pred["race_no"] = pred["race_id"].str.extract(r"(\d+)$")[0].astype(int)
    by = {k: g for k, g in pred.groupby(["date", "tk", "race_no"])}
    days = []
    for r in altili.itertuples():
        d = pd.Timestamp(r.race_date)
        legs = []
        for n in range(r.last_race_no - 5, r.last_race_no + 1):
            g = by.get((d, r.tk, n))
            if g is None or g["is_winner"].sum() < 1:
                legs = None
                break
            legs.append(g)
        if legs:
            days.append({"date": d, "payout": r.payout, "legs": legs})
    return days


def _score(day: dict, col: str, alloc_fn, budget: int) -> tuple[bool, int]:
    arrays = [g[col].to_numpy() for g in day["legs"]]
    ks = alloc_fn(arrays, budget)
    hit = True
    for g, k in zip(day["legs"], ks):
        top = g.sort_values(col, ascending=False).head(k)
        hit &= bool(top["is_winner"].max() == 1)
    return hit, int(np.prod(ks))


def evaluate(days: list[dict], seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    out: dict = {"n_altili": len(days), "n_days": len({d["date"] for d in days}), "budgets": {}}
    if not days:
        return out
    dates = np.array([d["date"] for d in days])
    udates = np.unique(dates)
    for b in BUDGETS:
        strategies = {
            "optimizer_model": ("p_cal", allocate), "greedy_market": ("q", allocate),
            "fixed_model": ("p_cal", fixed_alloc), "fixed_market": ("q", fixed_alloc),
        }
        res = {name: [_score(d, col, fn, b) for d in days] for name, (col, fn) in strategies.items()}
        hits = {n: np.array([h for h, _ in v], float) for n, v in res.items()}
        costs = {n: np.array([c for _, c in v], float) for n, v in res.items()}
        pay = np.array([d["payout"] or 0.0 for d in days])
        entry = {"strategies": {}}
        for n in strategies:
            roi = float((hits[n] * pay).sum() / costs[n].sum() - 1)
            entry["strategies"][n] = {"hit_rate": float(hits[n].mean()), "mean_cost": float(costs[n].mean()),
                                      "hits": int(hits[n].sum()), "published_unit_roi": roi}
        idx = [np.flatnonzero(dates == u) for u in udates]
        diffs = hits["optimizer_model"] - hits["greedy_market"]
        stats = []
        for _ in range(BOOT):
            pick = np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])
            stats.append(diffs[pick].mean())
        entry["delta_hit_vs_greedy_market"] = float(diffs.mean())
        entry["delta_ci95"] = [float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))]
        diffs2 = hits["optimizer_model"] - hits["fixed_market"]
        entry["delta_hit_vs_fixed_market"] = float(diffs2.mean())
        out["budgets"][str(b)] = entry
    legs = [(g, d) for d in days for g in d["legs"]]
    cov = {}
    for k in range(1, 7):
        row = {}
        for col in ("p_cal", "q"):
            pc, ac, af, al = [], [], [], []
            for g, _ in legs:
                top = g.sort_values(col, ascending=False).head(k)
                win = g[g["is_winner"] == 1]
                covered = float(top["is_winner"].max() == 1)
                pc.append(float(top["p_cal"].sum()) if col == "p_cal" else float(top["q"].sum()))
                ac.append(covered)
                (al if win["odds"].max() >= E.LONGSHOT_ODDS else af).append(covered)
            row[col] = {"predicted": float(np.mean(pc)), "actual": float(np.mean(ac)),
                        "fav_winner_cov": float(np.mean(af)) if af else None,
                        "longshot_winner_cov": float(np.mean(al)) if al else None, "n_longshot_winners": len(al)}
        cov[str(k)] = row
    out["coverage_by_k"] = cov
    return out


def run(features_csv: Path, altili_path: Path, cache: Path, final: bool = False) -> dict:
    df = E.prepare_frame(features_csv, cache / "edge_frame.pkl")
    all_days = sorted(df["date"].unique())
    preds = {}
    for name, fn in (("C1", E.make_cl(lambda d: E.base_features(df))), ("C2", E.make_gbm(lambda d: E.base_features(df)))):
        cp = cache / f"coupon_pred_{name}.pkl"
        if cp.exists() and cp.stat().st_mtime > (cache / "edge_frame.pkl").stat().st_mtime:
            preds[name] = pd.read_pickle(cp)
        else:
            preds[name] = E.walk_forward(df, fn, all_days, strict_only=False)
            preds[name].to_pickle(cp)
    altili = load_altili(altili_path)
    out: dict = {"protocol": "coupon-v1", "stage": "holdout" if final else "dev", "models": {}}
    for name, pred in preds.items():
        pred = calibrate_online(pred)
        days = build_days(pred, altili)
        dts = sorted({d["date"] for d in days})
        cut = dts[int(round(len(dts) * 0.8))] if dts else None
        subset = [d for d in days if (d["date"] >= cut if final else d["date"] < cut)] if cut is not None else []
        ev = evaluate(subset)
        ev["cut_date"] = str(cut)[:10] if cut is not None else None
        if not final:
            ev["logloss_calibrated"] = float(-np.log(pred.loc[pred["is_winner"] == 1, "p_cal"].clip(1e-9)).mean())
            ev["logloss_raw"] = float(-np.log(pred.loc[pred["is_winner"] == 1, "p"].clip(1e-9)).mean())
        out["models"][name] = ev
    return out


def replay(cache: Path, altili_path: Path, race_date: str | None, budget: int, model: str = "C1") -> dict:
    """Yerel veriden (walk-forward tahminleri) bir altili icin kupon onerisi; deneysel, canli oran gerektirmez."""
    pred = calibrate_online(pd.read_pickle(cache / f"coupon_pred_{model}.pkl"))
    altili = load_altili(altili_path)
    days = build_days(pred, altili)
    if not days:
        return {"altilis": [], "ticket": None}
    options = [f"{d['date'].date()}|{i}" for i, d in enumerate(days)]
    pick = next((d for d in reversed(days) if str(d["date"].date()) == race_date), days[-1])
    arrays = [g["p_cal"].to_numpy() for g in pick["legs"]]
    ks = allocate(arrays, budget)
    legs, cover, hit = [], 1.0, True
    for g, k in zip(pick["legs"], ks):
        top = g.sort_values("p_cal", ascending=False).head(k)
        cover *= float(top["p_cal"].sum())
        hit &= bool(top["is_winner"].max() == 1)
        legs.append([{"no": int(r.draw), "name": r.horse_name, "p": round(float(r.p_cal), 3),
                      "odds": float(r.odds), "won": bool(r.is_winner == 1)} for r in top.itertuples()])
    return {"date": str(pick["date"].date()), "dates": sorted({str(d["date"].date()) for d in days}),
            "ks": ks, "cost": int(np.prod(ks)), "model_hit_probability": cover, "hit": hit,
            "payout": pick["payout"], "legs": legs}


def save(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, default=float), encoding="utf-8")
