from __future__ import annotations

from datetime import date, timedelta
from collections.abc import Callable

import numpy as np
import pandas as pd

from atyaris.data_sources.base import DataSourceError
from atyaris.data_sources.tjk_scraper import TJKHtmlDataSource
from atyaris.models.entities import HorseStatistics
from atyaris.services import build_data_source


def _track_label(hippodrome: str) -> str:
    value = (hippodrome or "").strip().upper()
    return value.replace("İ", "I")


def _market_probability_from_odds(odds: float | None, default: float) -> float:
    if odds is None or odds <= 1.0:
        return default
    return 1.0 / odds


def ingest_real_tjk_data(
    start_date: date,
    end_date: date,
    paths,
    progress_callback: Callable[[str], None] | None = None,
) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    """Build the ML raw frame strictly from live TJK pages (no synthetic fallback)."""
    source = build_data_source("tjk", __import__("atyaris.config", fromlist=["get_settings"]).get_settings())
    if not isinstance(source, TJKHtmlDataSource):
        raise RuntimeError("ML egitimi icin gercek TJK kaynagi bekleniyor.")

    rows: list[dict[str, object]] = []
    current = start_date
    while current <= end_date:
        if progress_callback is not None:
            progress_callback(f"Veri cekiliyor: {current.isoformat()} (toplam satir: {len(rows)})")
        for hippodrome in source.get_available_hippodromes(current):
            if progress_callback is not None:
                progress_callback(f"  Hipodrom: {hippodrome}")
            races = source.get_daily_races(current, hippodrome)
            if not races:
                continue
            results = source.get_daily_race_results(current, hippodrome)

            for race in races:
                race_result = results.get(race.race_no, {})
                field_size = len(race.entries)
                if field_size <= 0:
                    continue

                for entry in race.entries:
                    finish_position = entry.actual_finish_position
                    if finish_position is None:
                        finish_position = race_result.get(entry.number)
                    if finish_position is None:
                        # Without a real race result, this row cannot be used for
                        # academic training/evaluation without inventing labels.
                        continue

                    try:
                        stats = source.get_horse_statistics(entry)
                    except DataSourceError as exc:
                        # TJK may omit an individual horse history while still
                        # exposing the race and its official result. Keep the
                        # labelled row and let feature defaults represent the
                        # missing history instead of aborting the whole run.
                        stats = HorseStatistics(
                            horse_id=str(entry.horse_id),
                            horse_name=str(entry.horse_name),
                        )
                        if progress_callback is not None:
                            progress_callback(
                                f"    Uyari: {entry.horse_name} gecmisi alinamadi; varsayilan istatistik kullaniliyor ({exc})"
                            )
                    history = [
                        p
                        for p in stats.past_performances
                        if p.race_date < current
                    ]
                    history.sort(key=lambda p: p.race_date)

                    perf_hist = [
                        1.0 - ((float(p.finish_position) - 1.0) / max(float(p.field_size or field_size) - 1.0, 1.0))
                        for p in history
                        if p.finish_position is not None
                    ]
                    recent_3 = perf_hist[-3:] if perf_hist else []
                    recent_5 = perf_hist[-5:] if perf_hist else []
                    recent_10 = perf_hist[-10:] if perf_hist else []

                    days_since_last = (
                        float((current - history[-1].race_date).days) if history else 30.0
                    )
                    last_distance = float(history[-1].distance_m) if history else float(race.distance_m)
                    distance_shock = abs(float(race.distance_m) - last_distance) / max(float(race.distance_m), 1.0)
                    last_perf = recent_3[-1] if recent_3 else 0.45
                    short_rest_flag = 1.0 if days_since_last < 8 else 0.0
                    long_layoff_flag = 1.0 if days_since_last > 75 else 0.0
                    seasonal_load = float(
                        sum(1 for p in history if (current - p.race_date).days <= 120)
                    ) / 120.0
                    race_frequency_3 = 3.0 / max(days_since_last, 1.0)
                    race_frequency_5 = 5.0 / max(days_since_last, 1.0)

                    fatigue_load = (
                        1.8 * short_rest_flag
                        + 0.6 * race_frequency_3
                        + 0.4 * race_frequency_5
                        + 0.8 * seasonal_load
                        + 0.5 * distance_shock
                        + 0.5 * (1.0 - last_perf)
                        + 0.4 * long_layoff_flag
                    )
                    recovery_score = float(1.0 / (1.0 + np.exp(fatigue_load - 1.5)))
                    fatigue_score = float(1.0 / (1.0 + np.exp(-(days_since_last - 18.0) / 8.0)))

                    pace_values = [
                        float(p.early_pace_index)
                        for p in history
                        if p.early_pace_index is not None
                    ]
                    pace_hint = float(pace_values[-1]) if pace_values else 0.0
                    style_front_prob = (
                        float(sum(1 for p in pace_values if p >= 0.6)) / len(pace_values)
                        if pace_values
                        else 0.25
                    )
                    style_presser_prob = (
                        float(sum(1 for p in pace_values if 0.2 <= p < 0.6)) / len(pace_values)
                        if pace_values
                        else 0.25
                    )
                    style_stalker_prob = (
                        float(sum(1 for p in pace_values if -0.25 <= p < 0.2)) / len(pace_values)
                        if pace_values
                        else 0.25
                    )
                    style_closer_prob = (
                        float(sum(1 for p in pace_values if p < -0.25)) / len(pace_values)
                        if pace_values
                        else 0.25
                    )

                    surface = race.surface.value if hasattr(race.surface, "value") else str(race.surface)
                    surface_fit = (
                        float(
                            sum(
                                perf
                                for perf, p in zip(perf_hist, history)
                                if (p.surface.value if hasattr(p.surface, "value") else str(p.surface)) == surface
                            )
                            / max(
                                1,
                                sum(
                                    1
                                    for p in history
                                    if (p.surface.value if hasattr(p.surface, "value") else str(p.surface)) == surface
                                ),
                            )
                        )
                        if history
                        else 0.45
                    )
                    track_fit = (
                        float(
                            sum(
                                perf
                                for perf, p in zip(perf_hist, history)
                                if str(p.hippodrome) == str(race.hippodrome)
                            )
                            / max(1, sum(1 for p in history if str(p.hippodrome) == str(race.hippodrome)))
                        )
                        if history
                        else 0.45
                    )
                    condition_fit = 0.45
                    distance_fit = (
                        float(
                            sum(
                                perf
                                for perf, p in zip(perf_hist, history)
                                if abs(float(p.distance_m) - float(race.distance_m)) <= 200.0
                            )
                            / max(
                                1,
                                sum(
                                    1
                                    for p in history
                                    if abs(float(p.distance_m) - float(race.distance_m)) <= 200.0
                                ),
                            )
                        )
                        if history
                        else 0.45
                    )

                    market_probability = _market_probability_from_odds(entry.odds, 1.0 / max(field_size, 1))
                    implied_probability = market_probability

                    rows.append(
                        {
                            "race_id": str(race.id),
                            "date": current,
                            "race_datetime": race.start_time,
                            "horse_id": str(entry.horse_id),
                            "horse_name": str(entry.horse_name),
                            "draw": int(entry.number),
                            "weight": float(entry.weight_kg),
                            "distance": float(race.distance_m),
                            "field_size": float(field_size),
                            "form_avg_3": float(sum(recent_3) / len(recent_3)) if recent_3 else 0.45,
                            "form_avg_5": float(sum(recent_5) / len(recent_5)) if recent_5 else 0.45,
                            "form_avg_10": float(sum(recent_10) / len(recent_10)) if recent_10 else 0.45,
                            "form_var_5": float(pd.Series(recent_5).var()) if len(recent_5) >= 2 else 0.03,
                            "last_run_perf": float(last_perf),
                            "trend_3_10": (
                                float(sum(recent_3) / len(recent_3)) if recent_3 else 0.45
                            )
                            - (
                                float(sum(recent_10) / len(recent_10)) if recent_10 else 0.45
                            ),
                            "days_since_last_race": days_since_last,
                            "fatigue_score": fatigue_score,
                            "recovery_score": recovery_score,
                            "short_rest_flag": short_rest_flag,
                            "long_layoff_flag": long_layoff_flag,
                            "race_frequency_3": race_frequency_3,
                            "race_frequency_5": race_frequency_5,
                            "seasonal_race_load": seasonal_load,
                            "pace_hint": pace_hint,
                            "style_front_prob": style_front_prob,
                            "style_presser_prob": style_presser_prob,
                            "style_stalker_prob": style_stalker_prob,
                            "style_closer_prob": style_closer_prob,
                            "distance_fit": distance_fit,
                            "surface_fit": surface_fit,
                            "track_fit": track_fit,
                            "condition_fit": condition_fit,
                            "market_probability": market_probability,
                            "implied_probability": implied_probability,
                            "odds": float(entry.odds) if entry.odds is not None else 0.0,
                            "is_winner": int(finish_position == 1),
                            "track": _track_label(race.hippodrome),
                            "surface": surface,
                            "track_condition": "NORMAL",
                        }
                    )

        current += timedelta(days=1)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(
            "Gercek TJK verisiyle ML egitimi icin kullanilabilir satir bulunamadi. "
            "Bulten/sonuc/at gecmisi scraping dogrulanmali."
        )

    required_columns = {
        "race_id",
        "date",
        "race_datetime",
        "horse_id",
        "draw",
        "weight",
        "distance",
        "field_size",
        "market_probability",
        "implied_probability",
        "odds",
        "is_winner",
    }
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise RuntimeError(f"Gercek TJK veri cercevesinde eksik zorunlu kolonlar var: {missing}")

    return frame
