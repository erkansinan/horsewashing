from __future__ import annotations

from datetime import date, timedelta
from collections.abc import Callable
from time import monotonic

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


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "hesaplaniyor"
    rounded = max(0, int(seconds))
    minutes, remaining_seconds = divmod(rounded, 60)
    if minutes:
        return f"{minutes} dk {remaining_seconds} sn"
    return f"{remaining_seconds} sn"


def _estimate_remaining(started_at: float, completed: int, total: int) -> str:
    if completed <= 0 or total <= completed:
        return _format_duration(0.0 if total <= completed else None)
    elapsed = monotonic() - started_at
    return _format_duration((elapsed / completed) * (total - completed))


def ingest_real_tjk_data(
    start_date: date,
    end_date: date,
    paths,
    progress_callback: Callable[[str], None] | None = None,
    require_results: bool = True,
) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    """Build labelled training or unlabelled prediction rows from live TJK pages."""
    source = build_data_source("tjk", __import__("atyaris.config", fromlist=["get_settings"]).get_settings())
    if not isinstance(source, TJKHtmlDataSource):
        raise RuntimeError("ML egitimi icin gercek TJK kaynagi bekleniyor.")

    rows: list[dict[str, object]] = []
    daily_races: list[tuple[date, object, list[object], dict[int, dict[int, int]]]] = []
    unavailable_days: list[str] = []
    unavailable_result_sets: list[str] = []
    started_at = monotonic()
    total_days = max((end_date - start_date).days + 1, 1)
    completed_days = 0

    # Collect the complete labelled window before reading horse histories. This
    # keeps result acquisition separate from the historical replay phase.
    current = start_date
    while current <= end_date:
        if progress_callback is not None:
            progress_callback(
                f"Sonuclar aliniyor: {current.isoformat()} "
                f"({completed_days + 1}/{total_days} gun) | "
                f"tahmini kalan: {_estimate_remaining(started_at, completed_days, total_days)}"
            )
        try:
            hippodromes = source.get_available_hippodromes(current)
        except DataSourceError as exc:
            unavailable_days.append(f"{current.isoformat()}: {exc}")
            if progress_callback is not None:
                progress_callback(
                    f"    Uyari: {current.isoformat()} icin hipodrom listesi alinamadi; "
                    f"gun atlanacak ({exc})"
                )
            completed_days += 1
            current += timedelta(days=1)
            continue

        for hippodrome in hippodromes:
            if progress_callback is not None:
                progress_callback(
                    f"Sonuc sayfasi okunuyor: {current.isoformat()} / {hippodrome} | "
                    f"toplanan yarislari: {len(daily_races)} | "
                    f"tahmini kalan: {_estimate_remaining(started_at, completed_days, total_days)}"
                )
            try:
                races = source.get_daily_races(current, hippodrome)
            except DataSourceError as exc:
                if progress_callback is not None:
                    progress_callback(
                        f"    Uyari: {current.isoformat()} / {hippodrome} bulteni alinamadi; "
                        f"hipodrom atlanacak ({exc})"
                    )
                continue
            if not races:
                continue
            if require_results:
                try:
                    results = source.get_daily_race_results(current, hippodrome)
                except DataSourceError as exc:
                    unavailable_result_sets.append(f"{current.isoformat()} / {hippodrome}: {exc}")
                    if progress_callback is not None:
                        progress_callback(
                            f"    Uyari: {current.isoformat()} / {hippodrome} sonuclari alinamadi; "
                            f"hipodrom atlanacak ({exc})"
                        )
                    continue
                if not results:
                    unavailable_result_sets.append(f"{current.isoformat()} / {hippodrome}: bos sonuc")
                    if progress_callback is not None:
                        progress_callback(
                            f"    Uyari: {current.isoformat()} / {hippodrome} icin sonuc tablosu bos; "
                            "hipodrom atlanacak"
                        )
                    continue
            else:
                results = {}
            daily_races.append((current, hippodrome, races, results))
        completed_days += 1
        current += timedelta(days=1)

    total_entries = sum(len(race.entries) for _, _, races, _ in daily_races for race in races)
    completed_entries = 0
    replay_started_at = monotonic()

    for current, hippodrome, races, results in daily_races:
            for race in races:
                race_result = results.get(race.race_no, {})
                field_size = len(race.entries)
                if field_size <= 0:
                    continue

                for entry in race.entries:
                    completed_entries += 1
                    if progress_callback is not None:
                        progress_callback(
                            f"Tarihsel istatistik hesaplaniyor: {current.isoformat()} / "
                            f"{hippodrome} / {race.race_no}. kosu / {entry.horse_name} | "
                            f"gunluk at: {completed_entries}/{total_entries} | "
                            f"gunluk uretilen satir: {len(rows)} | "
                            f"tahmini kalan: {_estimate_remaining(replay_started_at, completed_entries, total_entries)}"
                        )
                    finish_position = None if not require_results else entry.actual_finish_position
                    if finish_position is None and require_results:
                        finish_position = race_result.get(entry.number)
                    if require_results and finish_position is None:
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

    frame = pd.DataFrame(rows)
    if frame.empty:
        details = []
        if unavailable_days:
            details.append(f"veri alinamayan gun sayisi: {len(unavailable_days)}")
        if unavailable_result_sets:
            details.append(f"sonuc alinamayan hipodrom sayisi: {len(unavailable_result_sets)}")
        suffix = f" ({'; '.join(details)})" if details else ""
        purpose = "ML egitimi" if require_results else "ML tahmini"
        raise RuntimeError(
            f"Gercek TJK verisiyle {purpose} icin kullanilabilir satir bulunamadi. "
            f"Bulten/sonuc/at gecmisi scraping dogrulanmali{suffix}."
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
