from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from collections.abc import Callable
import json
from pathlib import Path
import re
from statistics import median
from time import monotonic

import numpy as np
import pandas as pd

from atyaris.config import Settings
from atyaris.ml.backtest import WalkForwardResult, walk_forward_backtest
from atyaris.ml.calibration import (
    apply_calibrator,
    apply_probability_floor,
    fit_calibrator,
    recover_collapsed_calibration,
    from_payload,
    smooth_race_probabilities,
    to_payload,
)
from atyaris.ml.ev_kelly import add_ev_kelly_columns
from atyaris.ml.features import (
    TJK_STAGE1_FEATURE_COLUMNS,
    FeatureBuildResult,
    assert_no_leakage_columns,
    build_leakage_safe_features,
    preprocess_dataset,
)
from atyaris.ml.harville import add_harville_columns
from atyaris.ml.market_blend import (
    BenterTwoStageArtifact,
    extract_market_reference_probability,
    fit_two_stage_benter,
    predict_form_probability,
    predict_two_stage_probability,
)
from atyaris.ml.model_health import run_model_health_checks
from atyaris.ml.modeling import load_phase3_artifact, save_phase3_artifact
from atyaris.ml.optimizer import optimize_ticket_portfolio
from atyaris.ml.real_ingestion import ingest_real_tjk_data


@dataclass
class Phase1Paths:
    raw_csv: Path = Path("data/raw/tjk_real_races.csv")
    clean_csv: Path = Path("data/processed/clean_races.csv")
    features_csv: Path = Path("data/processed/features_phase1.csv")
    prediction_features_csv: Path = Path("data/processed/prediction_features_phase1.csv")
    model_path: Path = Path("models/phase1_logreg.joblib")


def _apply_prediction_reliability(
    frame: pd.DataFrame,
    model_probability: np.ndarray,
    market_probability: np.ndarray,
) -> np.ndarray:
    """Shrink weakly evidenced predictions toward the race market baseline."""
    starts = pd.to_numeric(
        frame.get("career_starts", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)
    reliability = np.clip(starts / 5.0, 0.0, 1.0)
    workout_missing = pd.to_numeric(
        frame.get("workout_missing", pd.Series(1.0, index=frame.index)),
        errors="coerce",
    ).fillna(1.0).to_numpy(dtype=float)
    reliability *= np.where(workout_missing >= 0.5, 0.5, 1.0)
    blended = market_probability + reliability * (model_probability - market_probability)
    for race_id, indices in frame.groupby("race_id", sort=False).indices.items():
        del race_id
        total = float(np.sum(blended[indices]))
        if total > 0.0:
            blended[indices] = blended[indices] / total
    return blended


def paths_from_settings(settings: Settings) -> Phase1Paths:
    return Phase1Paths(
        raw_csv=Path(settings.phase1_raw_csv_path),
        clean_csv=Path(settings.phase1_clean_csv_path),
        features_csv=Path(settings.phase1_features_csv_path),
        prediction_features_csv=Path(settings.phase1_prediction_features_csv_path),
        model_path=Path(settings.phase1_model_path),
    )


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _training_checkpoint_path(paths: Phase1Paths) -> Path:
    return paths.raw_csv.with_suffix(".progress.json")


_TRAINING_COLLECTION_VERSION = 2


def _write_csv_atomically(frame: pd.DataFrame, path: Path) -> None:
    _ensure_parent(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_training_checkpoint(path: Path, start_date: date, end_date: date, completed_dates: set[date]) -> None:
    _ensure_parent(path)
    payload = {
        "collection_version": _TRAINING_COLLECTION_VERSION,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "completed_dates": sorted(day.isoformat() for day in completed_dates),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_training_checkpoint(path: Path, start_date: date, end_date: date) -> set[date]:
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("collection_version") != _TRAINING_COLLECTION_VERSION:
            return set()
        if payload.get("start_date") != start_date.isoformat() or payload.get("end_date") != end_date.isoformat():
            return set()
        return {date.fromisoformat(value) for value in payload.get("completed_dates", [])}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return set()


def _require_rows(frame: pd.DataFrame, stage: str) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(
            f"{stage} bos veri uretti. Secilen tarih araliginda etiketli TJK yarisi bulunamadi."
        )
    return frame


def _format_training_duration(seconds: float | None) -> str:
    if seconds is None:
        return "hesaplaniyor"
    rounded = max(0, int(seconds))
    minutes, remaining_seconds = divmod(rounded, 60)
    if minutes:
        return f"{minutes} dk {remaining_seconds} sn"
    return f"{remaining_seconds} sn"


_DAILY_PROGRESS_RE = re.compile(r"gunluk at:\s*(\d+)\s*/\s*(\d+)")


def ingest_real_data(
    start_date: date,
    end_date: date,
    paths: Phase1Paths,
    progress_callback: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    checkpoint_path = _training_checkpoint_path(paths)
    completed_dates = _load_training_checkpoint(checkpoint_path, start_date, end_date)
    existing = pd.DataFrame()
    if completed_dates and paths.raw_csv.exists():
        existing = pd.read_csv(paths.raw_csv)
    else:
        completed_dates = set()

    current = start_date
    total_days = max((end_date - start_date).days + 1, 1)
    completed_day_durations: list[float] = []
    active_day_started_at: float | None = None
    active_fraction = 0.0
    skipped_dates: set[date] = set()
    retry_targets: list[str] = []

    def report(message: str, active_day: bool = False) -> None:
        nonlocal active_fraction
        if progress_callback is None:
            return
        progress_match = _DAILY_PROGRESS_RE.search(message)
        if progress_match is not None:
            completed_entries = int(progress_match.group(1))
            total_entries = int(progress_match.group(2))
            active_fraction = completed_entries / max(total_entries, 1)
        completed_count = len(completed_dates)
        remaining_days = max(total_days - completed_count - (1 if active_day else 0), 0)
        typical_day_seconds = median(completed_day_durations[-5:]) if completed_day_durations else None
        active_day_seconds = None
        if active_day and active_day_started_at is not None:
            elapsed_active_day = max(monotonic() - active_day_started_at, 0.0)
            if active_fraction >= 0.10:
                active_day_seconds = elapsed_active_day / active_fraction
        projected_day_seconds = active_day_seconds or typical_day_seconds
        if projected_day_seconds is None:
            estimate = None
        else:
            active_remaining = 0.0
            if active_day:
                if active_day_seconds is not None:
                    active_remaining = max(active_day_seconds * (1.0 - active_fraction), 0.0)
                elif typical_day_seconds is not None:
                    active_remaining = typical_day_seconds
            estimate = active_remaining + (remaining_days * projected_day_seconds)
        progress_callback(
            f"{message} | tum egitim tahmini kalan: {_format_training_duration(estimate)}"
        )

    while current <= end_date:
        if current in completed_dates:
            report(f"Atlandi, daha once tamamlandi: {current.isoformat()}")
            current += timedelta(days=1)
            continue

        report(
            f"Gunluk veri checkpoint: {current.isoformat()} "
            f"({len(completed_dates) + 1}/{total_days})",
            active_day=True,
        )
        active_day_started_at = monotonic()
        active_fraction = 0.0
        day_progress = None
        if progress_callback is not None:
            day_progress = lambda message, day=current: report(  # noqa: E731
                f"{day.isoformat()} | {message}",
                active_day=True,
            )
        if progress_callback is None:
            day_data = ingest_real_tjk_data(current, current, paths)
        else:
            day_data = ingest_real_tjk_data(
                current,
                current,
                paths,
                progress_callback=day_progress,
            )
        day_retry_targets = [str(target) for target in day_data.attrs.get("retry_targets", [])]
        retry_targets.extend(day_retry_targets)
        if day_data.empty:
            skipped_dates.add(current)
            retry_targets.append(current.isoformat())
            report(
                f"Gun atlandi: {current.isoformat()} | kullanilabilir sonuc yok; "
                "checkpoint acik tutuldu, daha sonra tekrar denenebilir"
            )
            current += timedelta(days=1)
            continue
        day_duration = max(monotonic() - active_day_started_at, 0.0)
        existing = pd.concat([existing, day_data], ignore_index=True, sort=False)
        dedupe_columns = [column for column in ["date", "race_id", "horse_id", "draw"] if column in existing.columns]
        if dedupe_columns:
            existing = existing.drop_duplicates(subset=dedupe_columns, keep="last")
        _write_csv_atomically(existing, paths.raw_csv)
        if not day_retry_targets:
            completed_dates.add(current)
            completed_day_durations.append(day_duration)
        active_day_started_at = None
        active_fraction = 0.0
        _write_training_checkpoint(checkpoint_path, start_date, end_date, completed_dates)
        completion_label = "Gun tamamlandi" if not day_retry_targets else "Gun kismen tamamlandi"
        report(
            f"{completion_label}: {current.isoformat()} | "
            f"bu gun {len(day_data)} at | toplam biriken {len(existing)} at"
        )
        current += timedelta(days=1)

    existing.attrs["skipped_dates"] = sorted(day.isoformat() for day in skipped_dates)
    existing.attrs["retry_targets"] = retry_targets
    if existing.empty and retry_targets:
        return existing
    _require_rows(existing, "TJK veri cekme")
    return existing


def preprocess_raw(paths: Phase1Paths) -> pd.DataFrame:
    frame = pd.read_csv(paths.raw_csv)
    _require_rows(frame, "Ham veri")
    cleaned = preprocess_dataset(frame)
    _require_rows(cleaned, "Veri temizleme")
    _write_csv_atomically(cleaned, paths.clean_csv)
    return cleaned


def build_features(paths: Phase1Paths) -> FeatureBuildResult:
    frame = pd.read_csv(paths.clean_csv)
    _require_rows(frame, "Temiz veri")
    built = build_leakage_safe_features(frame)
    _require_rows(built.frame, "Feature uretimi")
    assert_no_leakage_columns(built.feature_columns)
    _write_csv_atomically(built.frame, paths.features_csv)
    return built


def prepare_prediction_features(
    target_date: date,
    paths: Phase1Paths,
    progress_callback: Callable[[str], None] | None = None,
    hippodrome: str | None = None,
    race_no: int | None = None,
) -> pd.DataFrame:
    """Create target-day features without adding unlabelled rows to training data."""
    ingestion_kwargs = {
        "progress_callback": progress_callback,
        "require_results": False,
    }
    if hippodrome:
        ingestion_kwargs["hippodrome"] = hippodrome
    if race_no is not None:
        ingestion_kwargs["race_no"] = race_no
    raw = ingest_real_tjk_data(target_date, target_date, paths, **ingestion_kwargs)
    historical = pd.read_csv(paths.clean_csv) if paths.clean_csv.exists() else pd.DataFrame()
    if not historical.empty and "date" in historical.columns:
        historical_dates = pd.to_datetime(historical["date"], errors="coerce").dt.date
        # The target day is prediction-only. Even if results were later written
        # to clean_races.csv, they must not become history for this prediction.
        historical = historical[historical_dates < target_date].copy()
    combined = pd.concat([historical, raw], ignore_index=True, sort=False)
    built = build_leakage_safe_features(combined, as_of_date=target_date)
    prediction = built.frame[pd.to_datetime(built.frame["date"]).dt.date == target_date].copy()
    _require_rows(prediction, "Tahmin feature uretimi")
    _write_csv_atomically(prediction, paths.prediction_features_csv)
    return prediction


def _benter_feature_columns(frame: pd.DataFrame) -> list[str]:
    missing = [column for column in TJK_STAGE1_FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"TJK feature verisinde eksik kolonlar var: {missing}")
    return TJK_STAGE1_FEATURE_COLUMNS.copy()


def _split_temporal_frames(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split labelled rows by chronological date into 70/15/15 partitions."""
    dates = sorted(pd.to_datetime(frame["date"]).dt.date.unique())
    if len(dates) < 3:
        raise ValueError("Zamansal split icin en az 3 farkli tarih gerekli")

    train_date_count = max(1, int(round(len(dates) * 0.70)))
    validation_date_count = max(1, int(round(len(dates) * 0.15)))
    if train_date_count + validation_date_count >= len(dates):
        raise ValueError("Zamansal split icin train/validation/test araliklari olusturulamadi")

    train_dates = set(dates[:train_date_count])
    validation_dates = set(dates[train_date_count:train_date_count + validation_date_count])
    test_dates = set(dates[train_date_count + validation_date_count:])
    date_values = pd.to_datetime(frame["date"]).dt.date
    train_df = frame[date_values.isin(train_dates)].copy()
    validation_df = frame[date_values.isin(validation_dates)].copy()
    test_df = frame[date_values.isin(test_dates)].copy()

    if train_df.empty or validation_df.empty or test_df.empty:
        raise ValueError("Zamansal split bos bir partition olusturdu")
    if not (max(train_dates) < min(validation_dates) < min(test_dates)):
        raise ValueError("Zamansal split tarih araliklari cakismiyor")
    if len(train_df) <= len(test_df):
        raise ValueError("Train satir sayisi test satir sayisindan buyuk olmali")
    return train_df, validation_df, test_df


def train_phase1_model(
    paths: Phase1Paths,
    holdout_days: int = 30,
    calibration_days: int = 21,
    calibration_method: str = "isotonic",
) -> BenterTwoStageArtifact:
    frame = pd.read_csv(paths.features_csv)
    _require_rows(frame, "Feature verisi")
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    feature_columns = _benter_feature_columns(frame)

    train_df, calibration_df, test_df = _split_temporal_frames(frame)

    artifact = fit_two_stage_benter(
        train_df,
        calibration_df,
        feature_columns,
        stage1_penalty="l2",
        stage1_regularization=0.05,
        stage2_penalty="l2",
        stage2_regularization=0.02,
        checkpoint_dir=paths.model_path.parent / f"{paths.model_path.stem}_checkpoint",
    )

    raw_cal = predict_two_stage_probability(artifact, calibration_df)
    calibrator = fit_calibrator(
        y_true=calibration_df["is_winner"].to_numpy(),
        raw_prob=raw_cal,
        method=calibration_method,
    )

    _ensure_parent(paths.model_path)
    save_phase3_artifact(artifact, to_payload(calibrator), str(paths.model_path))
    run_model_health_checks(
        artifact,
        train_df,
        calibration_df,
        test_df,
        feature_columns,
        output_path=paths.model_path.with_suffix(".health.json"),
    )
    checkpoint_dir = paths.model_path.parent / f"{paths.model_path.stem}_checkpoint"
    for checkpoint_file in checkpoint_dir.glob("*.npz"):
        checkpoint_file.unlink(missing_ok=True)
    try:
        checkpoint_dir.rmdir()
    except OSError:
        pass
    return artifact


def predict_for_date(
    paths: Phase1Paths,
    target_date: date,
    enable_ev: bool = True,
    ev_probability_threshold: float = 0.18,
    ev_min_edge: float = 0.03,
    ev_min_value: float = 0.02,
) -> pd.DataFrame:
    training_frame = pd.read_csv(paths.features_csv)
    prediction_frame = (
        pd.read_csv(paths.prediction_features_csv)
        if paths.prediction_features_csv.exists()
        else pd.DataFrame()
    )
    training_frame["date"] = pd.to_datetime(training_frame["date"], errors="coerce").dt.date
    prediction_frame["date"] = (
        pd.to_datetime(prediction_frame["date"], errors="coerce").dt.date
        if not prediction_frame.empty and "date" in prediction_frame.columns
        else pd.Series(dtype="object")
    )
    target_prediction = prediction_frame[prediction_frame["date"] == target_date].copy()
    if not target_prediction.empty:
        # A dedicated prediction snapshot takes precedence over any labelled
        # rows for the same day that may exist in the training feature file.
        training_frame = training_frame[training_frame["date"] != target_date].copy()
        frame = pd.concat([training_frame, target_prediction], ignore_index=True, sort=False)
    else:
        frame = pd.concat([training_frame, prediction_frame], ignore_index=True, sort=False)
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    day_df = frame[frame["date"] == target_date].copy()

    # No rows for the requested date: return an empty frame with expected output columns
    # instead of passing zero samples into model/scaler steps.
    if day_df.empty:
        out = day_df.copy()
        for col in [
            "raw_probability",
            "calibrated_probability",
            "place2_probability",
            "place3_probability",
            "top3_probability",
            "rank",
            "confidence",
        ]:
            if col not in out.columns:
                out[col] = pd.Series(dtype="float64")
        if enable_ev:
            for col in ["market_probability", "implied_probability", "edge", "ev", "kelly_fraction", "bet_decision"]:
                if col not in out.columns:
                    if col == "bet_decision":
                        out[col] = pd.Series(dtype="object")
                    else:
                        out[col] = pd.Series(dtype="float64")
        return out

    artifact, calibrator_payload = load_phase3_artifact(str(paths.model_path))
    calibrator = from_payload(calibrator_payload)

    out = day_df.copy()
    form_probability = predict_form_probability(artifact, out)
    market_probability = extract_market_reference_probability(out)
    out["form_probability"] = form_probability
    out["raw_probability"] = predict_two_stage_probability(artifact, out)
    raw_probability = out["raw_probability"].to_numpy()
    calibrated_probability = recover_collapsed_calibration(
        apply_calibrator(calibrator, raw_probability),
        raw_probability,
    )
    out["calibrated_probability"] = smooth_race_probabilities(
        calibrated_probability,
        out.groupby("race_id")["race_id"].transform("size").to_numpy(),
    )
    out["calibrated_probability"] = _apply_prediction_reliability(
        out,
        out["calibrated_probability"].to_numpy(dtype=float),
        market_probability,
    )

    # Race-level normalization.
    denom = out.groupby("race_id")["calibrated_probability"].transform("sum").replace(0.0, 1.0)
    out["calibrated_probability"] = out["calibrated_probability"] / denom
    out["market_probability_used"] = market_probability
    out["rank"] = out.groupby("race_id")["calibrated_probability"].rank(ascending=False, method="dense")
    out = add_harville_columns(out, win_col="calibrated_probability")

    if enable_ev:
        out = add_ev_kelly_columns(
            out,
            min_probability=ev_probability_threshold,
            min_edge=ev_min_edge,
            min_ev=ev_min_value,
            fractional_kelly=0.35,
            max_kelly_fraction=0.25,
        )
    else:
        out["edge"] = out["calibrated_probability"] - out["market_probability_used"]
        out["ev"] = out["calibrated_probability"] * out["odds"] - 1.0
        out["kelly_fraction"] = 0.0
        out["bet_decision"] = "NO_BET"

    # Confidence proxy: larger model-market divergence indicates stronger model conviction.
    out["confidence"] = (0.5 + np.abs(out["edge"]).clip(upper=0.5)).astype(float)
    return out


def run_phase1_backtest(
    paths: Phase1Paths,
    min_train_days: int = 90,
    calibration_days: int = 21,
    calibration_method: str = "isotonic",
    ev_probability_threshold: float = 0.18,
    ev_min_edge: float = 0.03,
    ev_min_value: float = 0.02,
) -> dict[str, object]:
    frame = pd.read_csv(paths.clean_csv)
    result: WalkForwardResult = walk_forward_backtest(
        frame,
        min_train_days=min_train_days,
        calibration_days=calibration_days,
        calibration_method=calibration_method,
        ev_probability_threshold=ev_probability_threshold,
        ev_min_edge=ev_min_edge,
        ev_min_value=ev_min_value,
    )
    payload = asdict(result)
    payload["fold_max_train_date"] = [d.isoformat() for d in result.fold_max_train_date]
    payload["fold_test_date"] = [d.isoformat() for d in result.fold_test_date]
    return payload


def optimize_for_date(
    paths: Phase1Paths,
    target_date: date,
    settings: Settings,
    budget: float | None = None,
    prediction: pd.DataFrame | None = None,
) -> dict[str, object]:
    pred = prediction if prediction is not None else predict_for_date(
        paths,
        target_date,
        enable_ev=True,
        ev_probability_threshold=settings.ev_probability_threshold,
        ev_min_edge=settings.ev_min_edge,
        ev_min_value=settings.ev_min_value,
    )
    return optimize_ticket_portfolio(
        pred,
        budget=(budget if budget is not None else settings.phase4_default_budget),
        unit_cost=settings.phase4_unit_cost,
        beam_width=settings.phase4_beam_width,
        top_per_leg=settings.phase4_top_per_leg,
        simulation_count=settings.phase4_simulation_count,
        base_payout=settings.phase4_base_payout,
        no_bet_ev_threshold=settings.phase4_no_bet_ev_threshold,
        min_confidence=settings.phase4_min_confidence,
        concentration_penalty=settings.phase4_concentration_penalty,
    )
