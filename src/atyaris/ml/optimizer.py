from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TicketColumn:
    column_id: int
    race_ids: list[str]
    horse_ids: list[str]
    probability: float
    market_probability: float
    estimated_payout: float
    ev: float
    edge: float
    confidence: float
    cost: float
    monte_carlo_hit_rate: float
    strategy: str


def _build_legs(pred: pd.DataFrame, max_legs: int = 6, top_per_leg: int = 4) -> list[tuple[str, pd.DataFrame]]:
    races = sorted(pred["race_id"].unique())[:max_legs]
    legs: list[tuple[str, pd.DataFrame]] = []
    for race_id in races:
        leg = pred[pred["race_id"] == race_id].copy()
        leg = leg.sort_values("calibrated_probability", ascending=False).head(top_per_leg)
        if leg.empty:
            continue
        legs.append((str(race_id), leg))
    return legs


def _combo_metrics(combo_rows: list[pd.Series], base_payout: float) -> dict[str, float]:
    p = 1.0
    mp = 1.0
    conf = []
    for row in combo_rows:
        p *= float(row["calibrated_probability"])
        mp *= max(float(row.get("market_probability_used", row.get("market_probability_norm", 0.001))), 1e-6)
        conf.append(float(row.get("confidence", 0.5)))

    payout = base_payout / max(mp, 1e-9)
    ev = p * payout - 1.0
    edge = p - mp
    confidence = float(np.mean(conf)) if conf else 0.0
    return {
        "probability": float(p),
        "market_probability": float(mp),
        "estimated_payout": float(payout),
        "ev": float(ev),
        "edge": float(edge),
        "confidence": confidence,
    }


def _beam_search(
    legs: list[tuple[str, pd.DataFrame]],
    beam_width: int,
    base_payout: float,
    objective_probability_weight: float,
    objective_ev_weight: float,
    objective_edge_weight: float,
    objective_confidence_weight: float,
) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = [{"rows": [], "score": 0.0}]

    for _, leg_df in legs:
        expanded: list[dict[str, Any]] = []
        for state in states:
            for _, row in leg_df.iterrows():
                rows = state["rows"] + [row]
                m = _combo_metrics(rows, base_payout)
                score = (
                    objective_probability_weight * np.log(max(m["probability"], 1e-12))
                    + objective_ev_weight * m["ev"]
                    + objective_edge_weight * m["edge"]
                    + objective_confidence_weight * m["confidence"]
                )
                expanded.append({"rows": rows, "score": float(score)})

        expanded.sort(key=lambda x: x["score"], reverse=True)
        states = expanded[:beam_width]

    final_candidates: list[dict[str, Any]] = []
    for s in states:
        rows = s["rows"]
        m = _combo_metrics(rows, base_payout)
        race_ids = [str(r["race_id"]) for r in rows]
        horse_ids = [str(r["horse_id"]) for r in rows]
        final_candidates.append(
            {
                "race_ids": race_ids,
                "horse_ids": horse_ids,
                "probability": m["probability"],
                "market_probability": m["market_probability"],
                "estimated_payout": m["estimated_payout"],
                "ev": m["ev"],
                "edge": m["edge"],
                "confidence": m["confidence"],
                "score": s["score"],
            }
        )
    return final_candidates


def _simulate_winners(legs: list[tuple[str, pd.DataFrame]], simulation_count: int, random_seed: int) -> list[list[str]]:
    rng = np.random.default_rng(random_seed)
    outcomes: list[list[str]] = []
    for _ in range(simulation_count):
        winners: list[str] = []
        for _, leg_df in legs:
            probs = leg_df["calibrated_probability"].to_numpy(dtype=float)
            probs = probs / max(probs.sum(), 1e-9)
            idx = int(rng.choice(np.arange(len(leg_df)), p=probs))
            winners.append(str(leg_df.iloc[idx]["horse_id"]))
        outcomes.append(winners)
    return outcomes


def _estimate_hit_rate(combo_horses: list[str], outcomes: list[list[str]]) -> float:
    if not outcomes:
        return 0.0
    hits = sum(1 for o in outcomes if o == combo_horses)
    return float(hits / len(outcomes))


def optimize_ticket_portfolio(
    pred: pd.DataFrame,
    budget: float,
    unit_cost: float = 1.0,
    beam_width: int = 80,
    top_per_leg: int = 4,
    simulation_count: int = 20000,
    base_payout: float = 120000.0,
    no_bet_ev_threshold: float = 0.0,
    min_confidence: float = 0.35,
    concentration_penalty: float = 0.15,
    objective_probability_weight: float = 1.0,
    objective_ev_weight: float = 0.03,
    objective_edge_weight: float = 2.0,
    objective_confidence_weight: float = 0.5,
    random_seed: int = 42,
) -> dict[str, Any]:
    legs = _build_legs(pred, max_legs=6, top_per_leg=top_per_leg)
    if len(legs) < 6:
        return {
            "columns": [],
            "summary": {
                "status": "NO_BET",
                "reason": "6 ayak icin yeterli yaris yok.",
                "budget": budget,
                "spent": 0.0,
                "column_count": 0,
            },
        }

    candidates = _beam_search(
        legs,
        beam_width=beam_width,
        base_payout=base_payout,
        objective_probability_weight=objective_probability_weight,
        objective_ev_weight=objective_ev_weight,
        objective_edge_weight=objective_edge_weight,
        objective_confidence_weight=objective_confidence_weight,
    )

    outcomes = _simulate_winners(legs, simulation_count=simulation_count, random_seed=random_seed)

    for c in candidates:
        c["monte_carlo_hit_rate"] = _estimate_hit_rate(c["horse_ids"], outcomes)

    filtered = [
        c
        for c in candidates
        if c["ev"] >= no_bet_ev_threshold and c["confidence"] >= min_confidence
    ]

    if not filtered:
        return {
            "columns": [],
            "summary": {
                "status": "NO_BET",
                "reason": "EV veya confidence esiklerini gecen kolon yok.",
                "budget": budget,
                "spent": 0.0,
                "column_count": 0,
            },
        }

    max_columns = int(budget // unit_cost)
    selected: list[dict[str, Any]] = []

    while filtered and len(selected) < max_columns:
        best_ix = 0
        best_score = -1e18
        for i, c in enumerate(filtered):
            overlap_penalty = 0.0
            if selected:
                overlaps = []
                for s in selected:
                    same = sum(1 for a, b in zip(c["horse_ids"], s["horse_ids"]) if a == b)
                    overlaps.append(same / 6.0)
                overlap_penalty = concentration_penalty * max(overlaps)

            score = (
                1.0 * c["probability"]
                + 0.02 * c["ev"]
                + 2.0 * c["edge"]
                + 2.5 * c["monte_carlo_hit_rate"]
                + 0.5 * c["confidence"]
                - overlap_penalty
            )
            if score > best_score:
                best_score = score
                best_ix = i

        selected.append(filtered.pop(best_ix))

    columns: list[TicketColumn] = []
    for i, c in enumerate(selected, start=1):
        if i <= max(1, len(selected) // 3):
            strategy = "Conservative"
        elif i <= max(2, (2 * len(selected)) // 3):
            strategy = "Balanced"
        else:
            strategy = "Aggressive"

        columns.append(
            TicketColumn(
                column_id=i,
                race_ids=c["race_ids"],
                horse_ids=c["horse_ids"],
                probability=float(c["probability"]),
                market_probability=float(c["market_probability"]),
                estimated_payout=float(c["estimated_payout"]),
                ev=float(c["ev"]),
                edge=float(c["edge"]),
                confidence=float(c["confidence"]),
                cost=float(unit_cost),
                monte_carlo_hit_rate=float(c["monte_carlo_hit_rate"]),
                strategy=strategy,
            )
        )

    spent = float(len(columns) * unit_cost)
    return {
        "columns": [
            {
                "column_id": c.column_id,
                "combination": {race_id: horse_id for race_id, horse_id in zip(c.race_ids, c.horse_ids)},
                "probability": c.probability,
                "market_probability": c.market_probability,
                "estimated_payout": c.estimated_payout,
                "ev": c.ev,
                "edge": c.edge,
                "confidence": c.confidence,
                "cost": c.cost,
                "monte_carlo_hit_rate": c.monte_carlo_hit_rate,
                "strategy": c.strategy,
            }
            for c in columns
        ],
        "summary": {
            "status": "OK" if columns else "NO_BET",
            "budget": float(budget),
            "spent": spent,
            "column_count": len(columns),
            "total_expected_value": float(sum(c.ev for c in columns)),
            "average_hit_rate": float(np.mean([c.monte_carlo_hit_rate for c in columns])) if columns else 0.0,
        },
    }
