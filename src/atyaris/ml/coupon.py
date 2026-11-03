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


def replay(cache: Path, altili_path: Path, race_date: str | None, budget: int, model: str = "C1", mode: str = "hit") -> dict:
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
    sets_by_leg = [np.argsort(-g["p_cal"].to_numpy())[:k] for g, k in zip(pick["legs"], ks)]
    expected = None
    if mode == "ev":
        table = agf_product_table(altili_path, altili)
        prior = table[table["date"] < pick["date"]]
        if len(prior) >= 40:
            fit = fit_payout(prior)
            sets_by_leg, expected = ev_sets(pick["legs"], budget, fit)
            ks = [len(s) for s in sets_by_leg]
    for g, idx in zip(pick["legs"], sets_by_leg):
        top = g.iloc[idx].sort_values("p_cal", ascending=False)
        cover *= float(top["p_cal"].sum())
        hit &= bool(top["is_winner"].max() == 1)
        legs.append([{"no": int(r.draw), "name": r.horse_name, "p": round(float(r.p_cal), 3),
                      "odds": float(r.odds), "won": bool(r.is_winner == 1)} for r in top.itertuples()])
    return {"date": str(pick["date"].date()), "dates": sorted({str(d["date"].date()) for d in days}),
            "ks": ks, "cost": int(np.prod(ks)), "model_hit_probability": cover, "hit": hit, "mode": mode,
            "expected_net_ev": expected,
            "payout": pick["payout"], "legs": legs}


def save(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, default=float), encoding="utf-8")


# ------------------------------------------------------------------ EV modu (v2)
def agf_product_table(results_path: Path, altili: pd.DataFrame) -> pd.DataFrame:
    agf: dict[tuple, float] = {}
    for line in results_path.open(encoding="utf-8"):
        if '"horse"' not in line:
            continue
        r = json.loads(line)
        if r.get("agf_pct") and r.get("csv_order") == 1:
            agf[(r["race_date"], _track_key(r["hippodrome"]), r["race_no"])] = r["agf_pct"] / 100.0
    rows = []
    for r in altili.itertuples():
        legs = range(r.last_race_no - 5, r.last_race_no + 1)
        vals = [agf.get((r.race_date, r.tk, n)) for n in legs]
        if r.payout and all(v for v in vals):
            rows.append({"date": pd.Timestamp(r.race_date), "key": (r.race_date, r.tk, r.last_race_no),
                         "log_agf_prod": float(np.sum(np.log(vals))), "log_payout": float(np.log(r.payout))})
    return pd.DataFrame(rows)


def fit_payout(t: pd.DataFrame) -> dict:
    x, y = t["log_agf_prod"].to_numpy(), t["log_payout"].to_numpy()
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    ss = ((y - y.mean()) ** 2).sum()
    return {"a": float(a), "b": float(b), "r2": float(1 - ((y - pred) ** 2).sum() / ss) if ss else 0.0, "n": int(len(t))}


def payout_quality(t: pd.DataFrame, exclude_from: pd.Timestamp | None) -> dict:
    """Holdout disi gunlerde zamansal %70/%30 bolme: out-of-sample R2 ve ortalama log sapma."""
    t = t[t["date"] < exclude_from].sort_values("date") if exclude_from is not None else t.sort_values("date")
    cut = int(len(t) * 0.7)
    tr, te = t.iloc[:cut], t.iloc[cut:]
    f = fit_payout(tr)
    pred = f["a"] + f["b"] * te["log_agf_prod"]
    err = te["log_payout"] - pred
    ss = ((te["log_payout"] - te["log_payout"].mean()) ** 2).sum()
    return {"fit_train": f, "n_test": int(len(te)), "oos_r2": float(1 - (err ** 2).sum() / ss),
            "mean_log_error": float(err.mean()), "median_abs_pct_error": float(np.median(np.abs(np.exp(err) - 1)))}


def ev_sets(legs: list[pd.DataFrame], budget: int, fit: dict) -> tuple[list[np.ndarray], float]:
    """EV = e^a * prod_i sum_h p_h * agf_h^b - maliyet; faktorize oldugu icin greedy net-EV artisi."""
    ws, order = [], []
    for g in legs:
        agf = g["agf_pct"].fillna(g["agf_pct"].min() if g["agf_pct"].notna().any() else 1.0).clip(lower=0.1).to_numpy() / 100
        w = g["p_cal"].to_numpy() * agf ** fit["b"]
        o = np.argsort(-w)
        ws.append(w[o])
        order.append(o)
    ks = [1] * len(legs)
    scale = float(np.exp(fit["a"]))

    def net(ks_):
        return scale * float(np.prod([ws[i][: ks_[i]].sum() for i in range(len(ks_))])) - float(np.prod(ks_))

    cur = net(ks)
    while True:
        best, best_val = None, cur
        for i in range(len(ks)):
            if ks[i] >= len(ws[i]):
                continue
            trial = ks.copy()
            trial[i] += 1
            if int(np.prod(trial)) > budget:
                continue
            v = net(trial)
            if v > best_val:
                best, best_val = i, v
        if best is None:
            break
        ks[best] += 1
        cur = best_val
    return [order[i][: ks[i]] for i in range(len(ks))], cur


def _sets_by_col(legs, budget, col, fn):
    ks = fn([g[col].to_numpy() for g in legs], budget)
    return [np.argsort(-g[col].to_numpy())[:k] for g, k in zip(legs, ks)]


def evaluate_ev(days: list[dict], payout_table: pd.DataFrame, seed: int = 5) -> dict:
    rng = np.random.default_rng(seed)
    pt = payout_table.sort_values("date")
    out: dict = {"budgets": {}}
    usable = []
    for d in days:
        prior = pt[pt["date"] < d["date"]]
        if len(prior) >= 40:
            usable.append((d, fit_payout(prior)))
    out["n_altili"] = len(usable)
    out["n_days"] = len({d["date"] for d, _ in usable})
    if not usable:
        return out
    dates = np.array([d["date"] for d, _ in usable])
    udates = np.unique(dates)
    idx = [np.flatnonzero(dates == u) for u in udates]
    for b in BUDGETS:
        rec = {n: {"hit": [], "cost": [], "ret": []} for n in ("ev", "hit_opt", "market", "agf")}
        for d, fit in usable:
            legs = d["legs"]
            choices = {
                "ev": ev_sets(legs, b, fit)[0],
                "hit_opt": _sets_by_col(legs, b, "p_cal", allocate),
                "market": _sets_by_col(legs, b, "q", allocate),
                "agf": _sets_by_col(legs, b, "agf_pct", allocate),
            }
            for n, sets in choices.items():
                hit = all(g.iloc[s]["is_winner"].max() == 1 for g, s in zip(legs, sets))
                cost = float(np.prod([len(s) for s in sets]))
                rec[n]["hit"].append(float(hit))
                rec[n]["cost"].append(cost)
                rec[n]["ret"].append(float(hit) * (d["payout"] or 0.0))
        arr = {n: {k: np.array(v) for k, v in r.items()} for n, r in rec.items()}

        def roi(n, pick):
            c = arr[n]["cost"][pick].sum()
            return (arr[n]["ret"][pick].sum() - c) / c

        allp = np.arange(len(usable))
        entry = {"strategies": {n: {"hit_rate": float(arr[n]["hit"].mean()), "mean_cost": float(arr[n]["cost"].mean()),
                                    "roi": float(roi(n, allp))} for n in arr}}
        boots = {n: [] for n in arr}
        dbs = []
        for _ in range(BOOT):
            pick = np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])
            for n in arr:
                boots[n].append(roi(n, pick))
            dbs.append(roi("ev", pick) - roi("market", pick))
        for n in arr:
            entry["strategies"][n]["roi_ci95"] = [float(np.quantile(boots[n], 0.025)), float(np.quantile(boots[n], 0.975))]
        entry["delta_roi_ev_vs_market"] = float(entry["strategies"]["ev"]["roi"] - entry["strategies"]["market"]["roi"])
        entry["delta_roi_ci95"] = [float(np.quantile(dbs, 0.025)), float(np.quantile(dbs, 0.975))]
        out["budgets"][str(b)] = entry
    return out


def run_v2(features_csv: Path, results_path: Path, cache: Path, registry_path: Path, final: bool) -> dict:
    df = E.prepare_frame(features_csv, cache / "edge_frame.pkl")
    pred = calibrate_online(pd.read_pickle(cache / "coupon_pred_C2.pkl"))
    if "agf_pct" not in pred.columns:
        pred = pred.merge(df[["race_id", "horse_id", "agf_pct"]], on=["race_id", "horse_id"], how="left")
    altili = load_altili(results_path)
    days = build_days(pred, altili)
    dts = sorted({d["date"] for d in days})
    if registry_path.exists():
        reg = json.loads(registry_path.read_text(encoding="utf-8"))
    else:
        old_cut = dts[int(round(len(dts) * 0.8))]
        used = [x for x in dts if x >= old_cut]
        fresh = [x for x in dts if x < old_cut]
        cut = int(round(len(fresh) * 0.8))
        reg = {"used_old_holdout_dates": [str(x.date()) for x in used], "dev_dates": [str(x.date()) for x in fresh[:cut]],
               "clean_holdout_dates": [str(x.date()) for x in fresh[cut:]]}
        registry_path.write_text(json.dumps(reg, indent=1), encoding="utf-8")
    table = agf_product_table(results_path, altili)
    first_hold = pd.Timestamp(reg["clean_holdout_dates"][0])
    target = set(reg["clean_holdout_dates"] if final else reg["dev_dates"])
    subset = [d for d in days if str(d["date"].date()) in target]
    res = evaluate_ev(subset, table)
    res.update({"stage": "holdout" if final else "dev", "n_altili_total_with_payout": int(len(table)),
                "payout_model_all": fit_payout(table), "payout_quality_non_holdout": payout_quality(table, first_hold),
                "trigger_300_met": len(subset) >= 300})
    return res
