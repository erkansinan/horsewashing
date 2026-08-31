"""FastAPI tabanli basit HTML arayuzu.

``atyaris web`` komutuyla baslatilir (bkz. ``atyaris/cli/app.py``). Ayni
``atyaris.services`` yardimci fonksiyonlarini ve ``PredictionEngine``'i CLI
ile paylasir; boylece iki arayuz arasinda is mantigi tekrarlanmaz.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import re
from types import SimpleNamespace
from urllib.parse import urlencode
from xml.sax.saxutils import escape

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

from atyaris.config import get_settings
from atyaris.data_sources.base import DataSourceError
from atyaris.data_sources.tjk_scraper import TJKHtmlDataSource
from atyaris.ml.pipeline import (
    build_features,
    ingest_synthetic,
    optimize_for_date,
    paths_from_settings,
    predict_for_date,
    preprocess_raw,
    train_phase1_model,
)
from atyaris.ml.modeling import load_phase3_artifact
from atyaris.prediction.engine import PredictionEngine
from atyaris.services import InvalidSourceError, build_data_source, fetch_races, parse_date
from atyaris.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)

_SORTABLE_FIELDS = {
    "strategy": "Stratejik Sira",
    "number": "No",
    "odds": "Ganyan",
    "total": "Toplam",
    "form": "Form",
    "jockey_trainer": "Jokey",
    "distance_surface": "Pist",
    "weight": "Agirlik",
    "rest": "Dinlenme",
    "win_probability": "Kazanma Olasiligi",
    "confidence": "Guven",
    "ev": "EV",
    "kelly": "Kelly",
}

_ML_SORTABLE_FIELDS = {
    "rank": "Rank",
    "horse_id": "At ID",
    "odds": "Ganyan",
    "calibrated_probability": "P(win)",
    "confidence": "Guven",
    "edge": "Edge",
    "ev": "EV",
}

_TRACK_DISPLAY_MAP = {
    "ANKARA": "Ankara",
    "ISTANBUL": "İstanbul",
    "IZMIR": "İzmir",
}

_TR_ASCII_MAP = str.maketrans(
    {
        "ç": "c",
        "Ç": "C",
        "ğ": "g",
        "Ğ": "G",
        "ı": "i",
        "İ": "I",
        "ö": "o",
        "Ö": "O",
        "ş": "s",
        "Ş": "S",
        "ü": "u",
        "Ü": "U",
    }
)

_TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


@dataclass
class MLRaceSummary:
    race_id: str
    race_name: str
    horse_count: int
    top_horse_name: str
    top_probability: float
    value_bet_count: int


def _bulletin_source_for_ml(settings):  # type: ignore[no-untyped-def]
    """ML web flow uses real daily bulletin for date/city/race selection."""
    return build_data_source("tjk", settings)


def _safe_metric(value: float | None, fallback: float) -> float:
    return value if value is not None else fallback


def _safe_float(value, default: float = 0.0) -> float:  # type: ignore[no-untyped-def]
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _prediction_sort_key(item, sort_by: str):  # type: ignore[no-untyped-def]
    if sort_by == "strategy":
        win_probability = _safe_metric(getattr(item, "win_probability", None), -1.0)
        confidence_score = _safe_metric(getattr(item, "confidence_score", None), -1.0)
        value_bet = getattr(item, "value_bet", None)
        expected_value = _safe_metric(
            getattr(value_bet, "expected_value", None) if value_bet else None,
            -9999.0,
        )
        kelly_stake = _safe_metric(
            getattr(value_bet, "fractional_kelly_stake", None) if value_bet else None,
            -9999.0,
        )
        score = getattr(item, "score", None)
        total_score = _safe_metric(getattr(score, "total_score", None) if score is not None else None, -1.0)
        return (win_probability, confidence_score, total_score, expected_value, kelly_stake)
    if sort_by == "number":
        return item.entry.number
    if sort_by == "odds":
        return item.entry.odds if item.entry.odds is not None else 9999.0
    if sort_by == "total":
        return item.score.total_score
    if sort_by == "form":
        return item.score.form_score
    if sort_by == "jockey_trainer":
        return item.score.jockey_trainer_score
    if sort_by == "distance_surface":
        return item.score.distance_surface_score
    if sort_by == "weight":
        return item.score.weight_score
    if sort_by == "rest":
        return item.score.rest_score
    if sort_by == "win_probability":
        return _safe_metric(getattr(item, "win_probability", None), -1.0)
    if sort_by == "confidence":
        return _safe_metric(getattr(item, "confidence_score", None), -1.0)
    if sort_by == "ev":
        value_bet = getattr(item, "value_bet", None)
        return _safe_metric(getattr(value_bet, "expected_value", None) if value_bet else None, -9999.0)
    if sort_by == "kelly":
        value_bet = getattr(item, "value_bet", None)
        return _safe_metric(getattr(value_bet, "fractional_kelly_stake", None) if value_bet else None, -9999.0)
    return item.score.total_score


def _ml_sort_value(row, sort_by: str):  # type: ignore[no-untyped-def]
    if sort_by == "rank":
        return int(row.get("rank", 9999))
    if sort_by == "horse_id":
        return str(row.get("horse_id", ""))
    if sort_by == "odds":
        return _safe_float(row.get("odds"), 9999.0)
    if sort_by == "calibrated_probability":
        return _safe_float(row.get("calibrated_probability"), -1.0)
    if sort_by == "confidence":
        return _safe_float(row.get("confidence"), -1.0)
    if sort_by == "edge":
        return _safe_float(row.get("edge"), -9999.0)
    if sort_by == "ev":
        return _safe_float(row.get("ev"), -9999.0)
    return _safe_float(row.get("calibrated_probability"), -1.0)


def _ml_race_summaries(frame: pd.DataFrame) -> list[MLRaceSummary]:
    if frame.empty:
        return []

    rows: list[MLRaceSummary] = []
    grouped = frame.sort_values(["race_id", "rank"]).groupby("race_id", sort=True)
    for race_id, race_df in grouped:
        first = race_df.iloc[0]
        rows.append(
            MLRaceSummary(
                race_id=str(race_id),
                race_name=str(first.get("race_name", race_id)),
                horse_count=int(len(race_df)),
                top_horse_name=str(first.get("horse_name", first.get("horse_id", "-"))),
                top_probability=float(first.get("calibrated_probability", 0.0)),
                value_bet_count=int((race_df.get("bet_decision", "NO_BET") == "BET").sum()),
            )
        )
    return rows


def _ml_race_summaries_from_races(  # type: ignore[no-untyped-def]
    races: list,
    data_source,
    settings,
) -> list[MLRaceSummary]:
    engine = PredictionEngine(data_source, settings)
    rows: list[MLRaceSummary] = []
    for race in sorted(races, key=lambda r: (r.hippodrome, r.race_no)):
        active = [e for e in race.entries if not e.is_scratched]
        top = active[0].horse_name if active else "-"
        top_probability = 0.0
        value_bet_count = 0
        try:
            race_pred = engine.predict(race)
            ranked = race_pred.ranked
            if ranked:
                top = ranked[0].entry.horse_name
                top_probability = _safe_float(ranked[0].win_probability, 0.0)
            value_bet_count = int(
                sum(1 for hp in ranked if hp.value_bet and hp.value_bet.is_value_bet)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("ML race summary prediction failed for %s: %s", race.id, exc)

        rows.append(
            MLRaceSummary(
                race_id=str(race.id),
                race_name=f"{race.hippodrome} - {race.race_no}. Kosu",
                horse_count=len(active),
                top_horse_name=top,
                top_probability=top_probability,
                value_bet_count=value_bet_count,
            )
        )
    return rows


def _parse_race_no(race_id: str) -> int | None:
    m = re.search(r"_(\d+)$", race_id)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _display_track_name(track: str) -> str:
    key = _normalize_track_key(track)
    return _TRACK_DISPLAY_MAP.get(key, track)


def _normalize_track_key(value: str) -> str:
    return value.strip().translate(_TR_ASCII_MAP).upper()


def _attach_ml_labels(paths, target_date: date, frame: pd.DataFrame) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    out = frame.copy()
    if out.empty:
        if "race_name" not in out.columns:
            out["race_name"] = pd.Series(dtype="object")
        if "horse_name" not in out.columns:
            out["horse_name"] = pd.Series(dtype="object")
        return out

    if not paths.clean_csv.exists():
        out["race_name"] = out["race_id"].astype(str)
        out["horse_name"] = out["horse_id"].astype(str)
        return out

    clean = pd.read_csv(paths.clean_csv)
    clean["date"] = pd.to_datetime(clean["date"]).dt.date
    day_clean = clean[clean["date"] == target_date].copy()
    if day_clean.empty:
        out["race_name"] = out["race_id"].astype(str)
        out["horse_name"] = out["horse_id"].astype(str)
        return out

    labels = day_clean[["race_id", "horse_id", "horse_name", "track"]].drop_duplicates()
    out = out.merge(labels, on=["race_id", "horse_id"], how="left", suffixes=("", "_label"))

    race_label_map: dict[str, str] = {}
    for rid in out["race_id"].astype(str).unique().tolist():
        sub = out[out["race_id"].astype(str) == rid]
        track = str(sub["track"].dropna().iloc[0]) if "track" in sub.columns and not sub["track"].dropna().empty else "-"
        track_display = _display_track_name(track)
        race_no = _parse_race_no(rid)
        if race_no is None:
            race_label_map[rid] = f"{track_display} - {rid}"
        else:
            race_label_map[rid] = f"{track_display} - {race_no}. Kosu"

    out["race_name"] = out["race_id"].astype(str).map(race_label_map)
    out["horse_name"] = out["horse_name"].fillna(out["horse_id"]).astype(str)
    return out


def _ml_has_date(paths, target_date: date) -> bool:  # type: ignore[no-untyped-def]
    if not paths.features_csv.exists():
        return False
    try:
        dates = pd.read_csv(paths.features_csv, usecols=["date"])
    except Exception:  # noqa: BLE001
        return False
    parsed = pd.to_datetime(dates["date"], errors="coerce").dt.date
    return bool((parsed == target_date).any())


def _ml_model_loadable(paths) -> bool:  # type: ignore[no-untyped-def]
    if not paths.model_path.exists():
        return False
    try:
        load_phase3_artifact(str(paths.model_path))
    except Exception:  # noqa: BLE001
        return False
    return True


def _ensure_ml_ready_for_date(settings, target_date: date) -> None:  # type: ignore[no-untyped-def]
    paths = paths_from_settings(settings)
    needs_refresh = (not _ml_model_loadable(paths)) or (not _ml_has_date(paths, target_date))
    if not needs_refresh:
        return

    start_date = target_date - timedelta(days=420)
    ingest_synthetic(start_date, target_date, paths)
    preprocess_raw(paths)
    build_features(paths)
    train_phase1_model(
        paths,
        holdout_days=settings.phase1_holdout_days,
        calibration_days=settings.phase3_calibration_days,
        calibration_method=settings.phase3_calibration_method,
    )


def _predict_ml_for_date_with_recovery(settings, target_date: date) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    paths = paths_from_settings(settings)
    _ensure_ml_ready_for_date(settings, target_date)
    try:
        pred = predict_for_date(
            paths,
            target_date,
            enable_ev=settings.phase3_enable_ev,
            ev_probability_threshold=settings.ev_probability_threshold,
            ev_min_edge=settings.ev_min_edge,
            ev_min_value=settings.ev_min_value,
        )
        return _attach_ml_labels(paths, target_date, pred)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ML predict failed, rebuilding artifacts once: %s", exc)
        # Force-refresh artifacts once in case persisted files are from an incompatible runtime.
        start_date = target_date - timedelta(days=420)
        ingest_synthetic(start_date, target_date, paths)
        preprocess_raw(paths)
        build_features(paths)
        train_phase1_model(
            paths,
            holdout_days=settings.phase1_holdout_days,
            calibration_days=settings.phase3_calibration_days,
            calibration_method=settings.phase3_calibration_method,
        )
        pred = predict_for_date(
            paths,
            target_date,
            enable_ev=settings.phase3_enable_ev,
            ev_probability_threshold=settings.ev_probability_threshold,
            ev_min_edge=settings.ev_min_edge,
            ev_min_value=settings.ev_min_value,
        )
        return _attach_ml_labels(paths, target_date, pred)


def create_app() -> FastAPI:
    """FastAPI uygulamasini olusturur (uvicorn factory olarak kullanilir)."""
    configure_logging()
    app = FastAPI(title="Turkiye At Yarisi Tahmin Araci", docs_url="/api/docs")

    @app.get("/", response_class=HTMLResponse)
    def index(
        request: Request,
        source: str = Query("sample", pattern="^(sample|tjk|ml)$"),
        date_str: str = Query("", alias="date"),
        city: str = Query(""),
    ) -> HTMLResponse:
        settings = get_settings()
        error = None
        info = None
        races = []
        ml_races: list[MLRaceSummary] = []
        hippodromes: list[str] = []
        resolved_date = date_str
        try:
            parsed_date = parse_date(date_str or None)
            resolved_date = parsed_date.isoformat()

            if source == "ml":
                data_source = _bulletin_source_for_ml(settings)
                if isinstance(data_source, TJKHtmlDataSource):
                    hippodromes = data_source.get_available_hippodromes(parsed_date)
                else:
                    all_races = fetch_races(data_source, parsed_date, None, None)
                    hippodromes = sorted({r.hippodrome for r in all_races})

                if city:
                    races = fetch_races(data_source, parsed_date, city, None)
                    ml_races = _ml_race_summaries_from_races(races, data_source, settings)
                else:
                    info = "Lutfen listeden bir hipodrom secin."
                    ml_races = []

                if not ml_races:
                    if city:
                        info = "Secili tarih ve hipodrom icin ML kosu bulunamadi."

                return templates.TemplateResponse(
                    request,
                    "index.html",
                    {
                        "races": races,
                        "ml_races": ml_races,
                        "error": error,
                        "info": info,
                        "source": source,
                        "date": resolved_date,
                        "city": city,
                        "hippodromes": hippodromes,
                    },
                )

            data_source = build_data_source(source, settings)
            if isinstance(data_source, TJKHtmlDataSource):
                hippodromes = data_source.get_available_hippodromes(parsed_date)
                if city:
                    races = fetch_races(data_source, parsed_date, city, None)
                    if not races:
                        info = (
                            "Secili hipodrom icin kosu listesi alinmadi. "
                            "TJK'nin JS/AJAX yapisi nedeniyle bu tarih icin veri gelmiyor olabilir."
                        )
                else:
                    races = []
                    info = "Lutfen listeden bir hipodrom secin."
            else:
                all_races = fetch_races(data_source, parsed_date, None, None)
                hippodromes = sorted({r.hippodrome for r in all_races})
                races = [r for r in all_races if not city or r.hippodrome == city]
        except (InvalidSourceError, ValueError) as exc:
            error = f"Gecersiz istek: {exc}"
        except DataSourceError as exc:
            error = f"Veri kaynagi hatasi: {exc}"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Index endpoint failure")
            error = f"Beklenmeyen hata: {exc}"

        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "races": races,
                "ml_races": ml_races,
                "error": error,
                "info": info,
                "source": source,
                "date": resolved_date,
                "city": city,
                "hippodromes": hippodromes,
            },
        )

    @app.get("/predict", response_class=HTMLResponse)
    def predict(
        request: Request,
        race_id: str,
        source: str = Query("sample", pattern="^(sample|tjk|ml)$"),
        date_str: str = Query("", alias="date"),
        city: str = Query(""),
        sort_by: str = Query(
            "strategy",
            pattern="^(strategy|number|odds|total|form|jockey_trainer|distance_surface|weight|rest|win_probability|confidence|ev|kelly|rank|horse_id|calibrated_probability|edge)$",
        ),
        sort_order: str = Query("desc", pattern="^(asc|desc)$"),
    ) -> HTMLResponse:
        settings = get_settings()
        error = None
        prediction = None
        ml_prediction = None
        display_rows = []
        sort_urls: dict[str, str] = {}
        pdf_url = ""
        table_usage_notes = [
            "1) Once Kazanma Olasiligi ve Guven ile guclu adaylari ayiklayin.",
            "2) Sonra EV > 0 olanlari value adayi olarak filtreleyin.",
            "3) Bahis buyuklugunu Kelly (fractional) degerine gore sinirlayin.",
        ]

        def _entry_key(entry) -> str:  # type: ignore[no-untyped-def]
            if entry.source_horse_id is not None:
                return f"source:{entry.source_horse_id}"
            return f"horse:{entry.horse_id}"

        def _build_predict_url(*, sort_field: str, sort_direction: str) -> str:
            return "/predict?" + urlencode(
                {
                    "race_id": race_id,
                    "source": source,
                    "date": date_str,
                    "city": city,
                    "sort_by": sort_field,
                    "sort_order": sort_direction,
                }
            )

        try:
            parsed_date = parse_date(date_str or None)

            if source == "ml":
                paths = paths_from_settings(settings)
                pred = _predict_ml_for_date_with_recovery(settings, parsed_date)
                race_df = pred[pred["race_id"].astype(str) == race_id].copy()

                if not race_df.empty:
                    ml_sort_by = sort_by if sort_by in _ML_SORTABLE_FIELDS else "calibrated_probability"
                    reverse = sort_order == "desc"
                    records = race_df.to_dict(orient="records")
                    records.sort(key=lambda r: _ml_sort_value(r, ml_sort_by), reverse=reverse)

                    horse_name_lookup = {
                        str(r.get("horse_id", "")): str(r.get("horse_name", r.get("horse_id", "")))
                        for r in records
                    }
                    race_name_lookup = {
                        str(r.get("race_id", "")): str(r.get("race_name", r.get("race_id", "")))
                        for r in records
                    }

                    opt = optimize_for_date(paths, parsed_date, settings, budget=settings.phase4_default_budget)
                    ticket_preview = []
                    for c in opt.get("columns", []):
                        combo = c.get("combination", {})
                        if race_id not in combo:
                            continue
                        combo_readable = {
                            race_name_lookup.get(str(k), str(k)): horse_name_lookup.get(str(v), str(v))
                            for k, v in combo.items()
                        }
                        ticket_preview.append({**c, "combination": combo_readable})

                    ml_prediction = {
                        "race_id": race_id,
                        "race_name": str(records[0].get("race_name", race_id)) if records else race_id,
                        "rows": records,
                        "summary": {
                            "horse_count": int(len(records)),
                            "bet_count": int(sum(1 for r in records if r.get("bet_decision") == "BET")),
                            "top_probability": _safe_float(records[0].get("calibrated_probability"), 0.0) if records else 0.0,
                        },
                        "ticket_preview": ticket_preview,
                        "optimization_summary": opt.get("summary", {}),
                        "disclaimer": "Bu tahminler istatistiksel analize dayanir, kesinlik tasimaz; sorumlu bahis oynayin.",
                    }

                    for field in _ML_SORTABLE_FIELDS:
                        next_direction = "asc" if field == ml_sort_by and sort_order == "desc" else "desc"
                        if field == ml_sort_by and sort_order == "asc":
                            next_direction = "desc"
                        sort_urls[field] = _build_predict_url(sort_field=field, sort_direction=next_direction)
                else:
                    # If race is from daily bulletin (e.g., TJK ids), generate ML-style table
                    # from PredictionEngine for the selected real race.
                    data_source = _bulletin_source_for_ml(settings)
                    if city:
                        races = fetch_races(data_source, parsed_date, city, None)
                    else:
                        races = fetch_races(data_source, parsed_date, None, None)
                    race = next((r for r in races if str(r.id) == str(race_id)), None)

                    if race is None:
                        error = "Secilen kosu bulunamadi. Lutfen listeden yeniden secin."
                    else:
                        engine = PredictionEngine(data_source, settings)
                        race_pred = engine.predict(race)

                        records = []
                        for i, hp in enumerate(race_pred.ranked, start=1):
                            ev = hp.value_bet.expected_value if hp.value_bet and hp.value_bet.expected_value is not None else None
                            edge = hp.value_bet.edge if hp.value_bet and hp.value_bet.edge is not None else None
                            decision = "BET" if hp.value_bet and hp.value_bet.is_value_bet else "NO_BET"
                            records.append(
                                {
                                    "rank": i,
                                    "number": hp.entry.number,
                                    "horse_id": hp.entry.horse_id,
                                    "horse_name": hp.entry.horse_name,
                                    "odds": hp.entry.odds,
                                    "calibrated_probability": hp.win_probability if hp.win_probability is not None else 0.0,
                                    "confidence": hp.confidence_score if hp.confidence_score is not None else 0.0,
                                    "edge": edge,
                                    "ev": ev,
                                    "bet_decision": decision,
                                    "race_id": str(race.id),
                                    "race_name": f"{race.hippodrome} - {race.race_no}. Kosu",
                                }
                            )

                        ml_sort_by = sort_by if sort_by in _ML_SORTABLE_FIELDS else "calibrated_probability"
                        reverse = sort_order == "desc"
                        records.sort(key=lambda r: _ml_sort_value(r, ml_sort_by), reverse=reverse)
                        for idx, rec in enumerate(records, start=1):
                            rec["rank"] = idx

                        ml_prediction = {
                            "race_id": str(race.id),
                            "race_name": f"{race.hippodrome} - {race.race_no}. Kosu",
                            "rows": records,
                            "summary": {
                                "horse_count": len(records),
                                "bet_count": int(sum(1 for r in records if r.get("bet_decision") == "BET")),
                                "top_probability": _safe_float(records[0].get("calibrated_probability"), 0.0) if records else 0.0,
                            },
                            "ticket_preview": [],
                            "optimization_summary": {
                                "status": "INFO",
                                "budget": settings.phase4_default_budget,
                                "spent": 0.0,
                                "column_count": 0,
                            },
                            "disclaimer": "Bu tahminler istatistiksel analize dayanir, kesinlik tasimaz; sorumlu bahis oynayin.",
                        }

                        for field in _ML_SORTABLE_FIELDS:
                            next_direction = "asc" if field == ml_sort_by and sort_order == "desc" else "desc"
                            if field == ml_sort_by and sort_order == "asc":
                                next_direction = "desc"
                            sort_urls[field] = _build_predict_url(sort_field=field, sort_direction=next_direction)

                return templates.TemplateResponse(
                    request,
                    "predict.html",
                    {
                        "prediction": prediction,
                        "ml_prediction": ml_prediction,
                        "rows": display_rows,
                        "pdf_url": pdf_url,
                        "table_usage_notes": table_usage_notes,
                        "error": error,
                        "source": source,
                        "date": date_str,
                        "city": city,
                        "sort_by": sort_by,
                        "sort_order": sort_order,
                        "sort_urls": sort_urls,
                        "sortable_fields": _ML_SORTABLE_FIELDS,
                    },
                )

            data_source = build_data_source(source, settings)
            if source == "tjk" and not city:
                error = "TJK kaynaginda tahmin uretmeden once bir hipodrom secin."
                return templates.TemplateResponse(
                    request,
                    "predict.html",
                    {
                        "prediction": prediction,
                        "ml_prediction": ml_prediction,
                        "rows": display_rows,
                        "table_usage_notes": table_usage_notes,
                        "error": error,
                        "source": source,
                        "date": date_str,
                        "city": city,
                        "sort_by": sort_by,
                        "sort_order": sort_order,
                        "sort_urls": sort_urls,
                        "sortable_fields": _SORTABLE_FIELDS,
                    },
                )

            races = fetch_races(data_source, parsed_date, city or None, None)
            race = next((r for r in races if r.id == race_id), None)
            if race is None:
                error = "Yaris bulunamadi (bulten degismis olabilir; lutfen tekrar secin)."
            else:
                engine = PredictionEngine(data_source, settings)
                prediction = engine.predict(race)
                reverse = sort_order == "desc"
                ranked_by_key = {_entry_key(hp.entry): hp for hp in prediction.ranked}
                active_rows = [
                    SimpleNamespace(
                        entry=hp.entry,
                        score=hp.score,
                        reasoning=hp.reasoning,
                        tag=hp.tag,
                        win_probability=hp.win_probability,
                        confidence_score=hp.confidence_score,
                        value_bet=hp.value_bet,
                        feature_snapshot=hp.feature_snapshot,
                        is_scratched=False,
                    )
                    for hp in prediction.ranked
                ]

                unscored_active_rows = [
                    SimpleNamespace(
                        entry=entry,
                        score=None,
                        reasoning=["Bu at aktif durumda; ancak istatistik verisi alinamadigi icin skorlanamadi."],
                        tag="Veri Eksik",
                        win_probability=None,
                        confidence_score=None,
                        value_bet=None,
                        feature_snapshot={},
                        is_scratched=False,
                    )
                    for entry in race.entries
                    if not entry.is_scratched and _entry_key(entry) not in ranked_by_key
                ]

                scratched_rows = [
                    SimpleNamespace(
                        entry=entry,
                        score=None,
                        reasoning=["Bu at kosmaz (scratch) olarak isaretli."],
                        tag="Koşmaz",
                        win_probability=None,
                        confidence_score=None,
                        value_bet=None,
                        feature_snapshot={},
                        is_scratched=True,
                    )
                    for entry in race.entries
                    if entry.is_scratched
                ]

                if sort_by == "number":
                    display_rows = sorted(
                        active_rows + unscored_active_rows + scratched_rows,
                        key=lambda row: _prediction_sort_key(row, sort_by),
                        reverse=reverse,
                    )
                else:
                    display_rows = sorted(
                        active_rows,
                        key=lambda row: _prediction_sort_key(row, sort_by),
                        reverse=reverse,
                    ) + sorted(
                        unscored_active_rows, key=lambda row: row.entry.number
                    ) + sorted(
                        scratched_rows, key=lambda row: row.entry.number
                    )

                pdf_url = "/predict-all-pdf?" + urlencode(
                    {
                        "source": source,
                        "date": date_str,
                        "city": city,
                        "sort_by": sort_by,
                        "sort_order": sort_order,
                    }
                )

                for field in _SORTABLE_FIELDS:
                    next_direction = "asc" if field == sort_by and sort_order == "desc" else "desc"
                    if field == sort_by and sort_order == "asc":
                        next_direction = "desc"
                    sort_urls[field] = _build_predict_url(sort_field=field, sort_direction=next_direction)
        except (InvalidSourceError, ValueError) as exc:
            error = f"Gecersiz istek: {exc}"
        except RuntimeError as exc:
            error = (
                f"Tahmin uretilemedi: {exc} "
                "(TJK kaynaginda at istatistikleri su an sinirli/erisilemez olabilir)."
            )
        except DataSourceError as exc:
            error = f"Veri kaynagi hatasi: {exc}"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Predict endpoint failure")
            error = f"Beklenmeyen hata: {exc}"

        return templates.TemplateResponse(
            request,
            "predict.html",
            {
                "prediction": prediction,
                "ml_prediction": ml_prediction,
                "rows": display_rows,
                "pdf_url": pdf_url,
                "table_usage_notes": table_usage_notes,
                "error": error,
                "source": source,
                "date": date_str,
                "city": city,
                "sort_by": sort_by,
                "sort_order": sort_order,
                "sort_urls": sort_urls,
                "sortable_fields": _SORTABLE_FIELDS,
            },
        )

    @app.get("/predict-all-pdf")
    def predict_all_pdf(
        source: str = Query("sample", pattern="^(sample|tjk|ml)$"),
        date_str: str = Query("", alias="date"),
        city: str = Query(""),
        sort_by: str = Query(
            "strategy",
            pattern="^(strategy|number|odds|total|form|jockey_trainer|distance_surface|weight|rest|win_probability|confidence|ev|kelly)$",
        ),
        sort_order: str = Query("desc", pattern="^(asc|desc)$"),
    ) -> Response:
        settings = get_settings()
        if source == "ml":
            raise HTTPException(status_code=400, detail="ML modunda PDF ciktisi simdilik desteklenmiyor.")
        if not city:
            raise HTTPException(status_code=400, detail="PDF olusturmak icin once bir hipodrom secin.")

        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4, landscape
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.lib.units import mm
            import reportlab
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            from reportlab.platypus import LongTable, Paragraph, SimpleDocTemplate, Spacer, TableStyle
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=400,
                detail="PDF olusturma icin 'reportlab' kurulu olmali. Lutfen bagimliliklari guncelleyin.",
            ) from exc

        parsed_date = parse_date(date_str or None)
        data_source = build_data_source(source, settings)
        races = fetch_races(data_source, parsed_date, city or None, None)
        if not races:
            raise HTTPException(
                status_code=400,
                detail="Secili tarih/hipodrom icin yaris bulunamadi; PDF olusturulamadi.",
            )

        engine = PredictionEngine(data_source, settings)
        from io import BytesIO

        def _resolve_pdf_font_paths() -> tuple[Path | None, Path | None]:
            candidates = [
                (
                    Path(reportlab.__file__).resolve().parent / "fonts" / "Vera.ttf",
                    Path(reportlab.__file__).resolve().parent / "fonts" / "VeraBd.ttf",
                ),
                (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
                (Path("C:/Windows/Fonts/segoeui.ttf"), Path("C:/Windows/Fonts/segoeuib.ttf")),
                (
                    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
                ),
                (Path("/Library/Fonts/Arial.ttf"), Path("/Library/Fonts/Arial Bold.ttf")),
                Path("C:/Windows/Fonts/arial.ttf"),
                Path("C:/Windows/Fonts/segoeui.ttf"),
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                Path("/Library/Fonts/Arial.ttf"),
            ]
            for candidate in candidates:
                if isinstance(candidate, tuple):
                    regular, bold = candidate
                    if regular.exists() and bold.exists():
                        return regular, bold
                elif candidate.exists():
                    return candidate, None
            return None, None

        body_font_name = "Helvetica"
        header_font_name = "Helvetica-Bold"
        body_font_path, header_font_path = _resolve_pdf_font_paths()
        fallback_ascii_pdf = body_font_path is None
        if body_font_path is not None:
            body_font_name = "TurkishSans"
            if body_font_name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(body_font_name, str(body_font_path)))
            if header_font_path is not None:
                header_font_name = "TurkishSansBold"
                if header_font_name not in pdfmetrics.getRegisteredFontNames():
                    pdfmetrics.registerFont(TTFont(header_font_name, str(header_font_path)))
            else:
                header_font_name = body_font_name

        _pdf_ascii_map = str.maketrans(
            {
                "ç": "c",
                "Ç": "C",
                "ğ": "g",
                "Ğ": "G",
                "ı": "i",
                "İ": "I",
                "ö": "o",
                "Ö": "O",
                "ş": "s",
                "Ş": "S",
                "ü": "u",
                "Ü": "U",
            }
        )

        def _pdf_text(value: str) -> str:
            text = value.translate(_pdf_ascii_map) if fallback_ascii_pdf else value
            return escape(text)

        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=landscape(A4),
            leftMargin=12 * mm,
            rightMargin=12 * mm,
            topMargin=12 * mm,
            bottomMargin=12 * mm,
        )
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "TitleTurkish",
            parent=styles["Heading2"],
            fontName=header_font_name,
            fontSize=13,
            leading=16,
        )
        info_style = ParagraphStyle(
            "InfoTurkish",
            parent=styles["Normal"],
            fontName=body_font_name,
            fontSize=9,
            leading=12,
        )
        race_style = ParagraphStyle(
            "RaceTurkish",
            parent=styles["Normal"],
            fontName=header_font_name,
            fontSize=10,
            leading=13,
        )
        cell_style = ParagraphStyle(
            "CellTurkish",
            parent=styles["Normal"],
            fontName=body_font_name,
            fontSize=8,
            leading=10,
        )
        cell_numeric_style = ParagraphStyle(
            "CellNumericTurkish",
            parent=cell_style,
            alignment=2,
        )
        header_cell_style = ParagraphStyle(
            "HeaderCellTurkish",
            parent=cell_style,
            fontName=header_font_name,
            fontSize=8.5,
            leading=10.5,
            alignment=1,
        )

        elements = [
            Paragraph(
                _pdf_text(f"Tahmin Raporu - {city} - {parsed_date.strftime('%d.%m.%Y')}"),
                title_style,
            ),
            Paragraph(_pdf_text(f"Kaynak: {source}"), info_style),
            Spacer(1, 6),
        ]

        generated_race_count = 0
        skipped_races: list[str] = []

        for race in races:
            try:
                prediction = engine.predict(
                    race,
                    include_backtest=False,
                    include_detailed_reasoning=False,
                )
            except (ValueError, RuntimeError, DataSourceError) as exc:
                skipped_races.append(f"{race.race_no}. kosu: {exc}")
                logger.warning(
                    "PDF olusturulurken %s %s. kosu atlandi: %s",
                    race.hippodrome,
                    race.race_no,
                    exc,
                )
                continue

            generated_race_count += 1
            elements.append(
                Paragraph(
                    _pdf_text(
                        f"Koşu {race.race_no} ({race.start_time.strftime('%H:%M')}) - "
                        f"{race.distance_m}m {race.surface.value}"
                    ),
                    race_style,
                )
            )

            reverse = sort_order == "desc"
            ranked = sorted(
                prediction.ranked,
                key=lambda hp: _prediction_sort_key(hp, sort_by),
                reverse=reverse,
            )
            rows = [
                [
                    Paragraph("No", header_cell_style),
                    Paragraph("At", header_cell_style),
                    Paragraph("Ganyan", header_cell_style),
                    Paragraph("K.Olas.", header_cell_style),
                    Paragraph("Guven", header_cell_style),
                    Paragraph("EV", header_cell_style),
                    Paragraph("Kelly", header_cell_style),
                    Paragraph("Toplam", header_cell_style),
                    Paragraph("Etiket", header_cell_style),
                ]
            ]
            for hp in ranked:
                odds_text = f"{hp.entry.odds:.2f}" if hp.entry.odds is not None else "-"
                prob_text = f"{(hp.win_probability or 0.0) * 100:.2f}%" if hp.win_probability is not None else "-"
                conf_text = f"{hp.confidence_score:.2f}" if hp.confidence_score is not None else "-"
                ev_text = (
                    f"{hp.value_bet.expected_value:.3f}"
                    if hp.value_bet and hp.value_bet.expected_value is not None
                    else "-"
                )
                kelly_text = (
                    f"{hp.value_bet.fractional_kelly_stake:.3f}"
                    if hp.value_bet and hp.value_bet.fractional_kelly_stake is not None
                    else "-"
                )
                rows.append(
                    [
                        Paragraph(str(hp.entry.number), cell_numeric_style),
                        Paragraph(_pdf_text(hp.entry.horse_name), cell_style),
                        Paragraph(odds_text, cell_numeric_style),
                        Paragraph(prob_text, cell_numeric_style),
                        Paragraph(conf_text, cell_numeric_style),
                        Paragraph(ev_text, cell_numeric_style),
                        Paragraph(kelly_text, cell_numeric_style),
                        Paragraph(f"{hp.score.total_score:.1f}", cell_numeric_style),
                        Paragraph(_pdf_text(hp.tag), cell_style),
                    ]
                )

            table = LongTable(
                rows,
                colWidths=[10 * mm, 58 * mm, 16 * mm, 18 * mm, 15 * mm, 14 * mm, 14 * mm, 15 * mm, 30 * mm],
                repeatRows=1,
            )
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 4),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]
                )
            )
            elements.append(table)
            elements.append(Spacer(1, 8))

        if generated_race_count == 0:
            detail = "Secili tarih/hipodrom icin PDF uretilemedi; kosular tahminlenemedi."
            if skipped_races:
                detail = f"{detail} Ilk hata: {skipped_races[0]}"
            raise HTTPException(status_code=400, detail=detail)

        doc.build(elements)
        pdf_bytes = buffer.getvalue()
        buffer.close()

        file_name = f"tahmin-raporu-{city}-{parsed_date.isoformat()}.pdf".replace(" ", "_")
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{file_name}"'},
        )

    return app
