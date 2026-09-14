from __future__ import annotations

"""Training-time sanity checks for the Benter pipeline.

The functions in this module are deliberately side-effect free except for the
optional JSON report written by ``run_model_health_checks``. They are usable as
small tests in the test suite and as diagnostics during a real training run.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atyaris.ml.fundamental_model import fit_conditional_logit, predict_conditional_logit_probability
from atyaris.ml.market_blend import BenterTwoStageArtifact, predict_form_probability, predict_two_stage_probability


POST_RACE_TOKENS = (
    "finish_position",
    "is_winner",
    "result",
    "finish_time",
    "final_time",
    "closing_odds",
    "post_race",
    "payout",
    "prize",
)

FEATURE_AVAILABILITY = {
    "draw": "race declaration",
    "weight": "race declaration",
    "distance": "race declaration",
    "field_size": "race declaration",
    "market_probability_norm": "pre-race market odds",
    "implied_probability": "pre-race market odds",
    "form_avg_3": "historical performances before race",
    "form_avg_5": "historical performances before race",
    "form_avg_10": "historical performances before race",
    "days_since_last_race": "historical performances before race",
}


@dataclass
class HealthCheckResult:
    name: str
    passed: bool
    details: dict[str, Any]


def _result(name: str, passed: bool, **details: Any) -> HealthCheckResult:
    return HealthCheckResult(name=name, passed=bool(passed), details=details)


def test_no_leakage(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> HealthCheckResult:
    """Check temporal ordering, post-race columns, and horse group separation."""
    train_max = pd.to_datetime(train_df["date"]).max()
    validation_min = pd.to_datetime(validation_df["date"]).min() if not validation_df.empty else train_max
    test_min = pd.to_datetime(test_df["date"]).min() if not test_df.empty else validation_min
    temporal_ok = train_max < validation_min and validation_min <= test_min

    leaked = sorted(
        col for col in feature_columns
        if any(token in col.lower() for token in POST_RACE_TOKENS)
    )
    train_horses = set(train_df.get("horse_id", pd.Series(dtype=str)).astype(str))
    validation_horses = set(validation_df.get("horse_id", pd.Series(dtype=str)).astype(str))
    test_horses = set(test_df.get("horse_id", pd.Series(dtype=str)).astype(str))
    group_ok = not (train_horses & validation_horses or train_horses & test_horses or validation_horses & test_horses)
    passed = bool(temporal_ok and not leaked and group_ok)
    return _result(
        "test_no_leakage",
        passed,
        train_max_date=str(train_max.date()) if not pd.isna(train_max) else None,
        validation_min_date=str(validation_min.date()) if not pd.isna(validation_min) else None,
        test_min_date=str(test_min.date()) if not pd.isna(test_min) else None,
        leaked_features=leaked,
        overlapping_horses={
            "train_validation": sorted(train_horses & validation_horses),
            "train_test": sorted(train_horses & test_horses),
            "validation_test": sorted(validation_horses & test_horses),
        },
    )


def feature_availability_report(feature_columns: list[str]) -> dict[str, dict[str, str | bool]]:
    """Answer whether each configured feature exists before the race starts."""
    return {
        column: {
            "available_before_race": not any(token in column.lower() for token in POST_RACE_TOKENS),
            "source": FEATURE_AVAILABILITY.get(column, "derived pre-race feature"),
        }
        for column in feature_columns
    }


def test_test_set_isolation(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> HealthCheckResult:
    leaked = [column for column in feature_columns if column not in train_df.columns or column not in validation_df.columns]
    disjoint_rows = set(train_df.index).isdisjoint(test_df.index) and set(validation_df.index).isdisjoint(test_df.index)
    return _result("test_test_set_isolation", bool(not leaked and disjoint_rows), missing_training_columns=leaked, row_indices_disjoint=disjoint_rows)


def _race_top1(frame: pd.DataFrame, probabilities: np.ndarray) -> float:
    if frame.empty:
        return 0.0
    work = frame[["race_id", "is_winner"]].copy()
    work["probability"] = probabilities
    picks = work.loc[work.groupby("race_id")["probability"].idxmax()]
    return float(picks["is_winner"].mean())


def _log_loss(frame: pd.DataFrame, probabilities: np.ndarray) -> float:
    if frame.empty:
        return 0.0
    p = np.clip(probabilities, 1e-8, 1.0 - 1e-8)
    y = frame["is_winner"].to_numpy(dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def test_beats_baseline(
    artifact: BenterTwoStageArtifact,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> HealthCheckResult:
    """Compare the full model with market-only and always-favorite baselines."""
    if test_df.empty:
        return _result("test_beats_baseline", False, reason="empty test set")
    market_col = "market_probability_norm" if "market_probability_norm" in train_df else "market_probability"
    market_model = fit_conditional_logit(train_df, [market_col], max_iter=80, random_state=42)
    market_p = predict_conditional_logit_probability(market_model, test_df)
    full_p = predict_two_stage_probability(artifact, test_df)
    market_top1 = _race_top1(test_df, market_p)
    full_top1 = _race_top1(test_df, full_p)
    favorite_top1 = _race_top1(test_df, test_df[market_col].to_numpy(dtype=float))
    return _result(
        "test_beats_baseline",
        bool(full_top1 > market_top1 and full_top1 > favorite_top1),
        full_log_loss=_log_loss(test_df, full_p),
        market_only_log_loss=_log_loss(test_df, market_p),
        full_top1=full_top1,
        market_only_top1=market_top1,
        favorite_top1=favorite_top1,
    )


def test_label_shuffle_fails(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    random_state: int = 42,
) -> HealthCheckResult:
    """Train on shuffled labels; performance must remain near race-level chance."""
    if train_df.empty or test_df.empty:
        return _result("test_label_shuffle_fails", False, reason="empty split")
    shuffled = train_df.copy()
    shuffled["is_winner"] = np.random.default_rng(random_state).permutation(shuffled["is_winner"].to_numpy())
    model = fit_conditional_logit(shuffled, feature_columns, max_iter=80, random_state=random_state)
    probabilities = predict_conditional_logit_probability(model, test_df)
    chance = float(1.0 / max(test_df.groupby("race_id").size().mean(), 1.0))
    top1 = _race_top1(test_df, probabilities)
    passed = top1 <= chance + 0.15
    return _result("test_label_shuffle_fails", passed, shuffled_top1=top1, chance_level=chance, tolerance=0.15)


def test_prediction_diversity(artifact: BenterTwoStageArtifact, frame: pd.DataFrame) -> HealthCheckResult:
    probabilities = predict_two_stage_probability(artifact, frame)
    variance = float(np.var(probabilities)) if probabilities.size else 0.0
    race_means = frame.assign(_p=probabilities).groupby("race_id")["_p"].mean() if probabilities.size else pd.Series(dtype=float)
    return _result(
        "test_prediction_diversity",
        bool(probabilities.size > 0 and variance > 1e-10 and race_means.nunique() > 1),
        probability_variance=variance,
        race_mean_count=int(race_means.nunique()),
    )


def test_reproducibility(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    random_state: int = 42,
) -> HealthCheckResult:
    first = fit_conditional_logit(train_df, feature_columns, max_iter=80, random_state=random_state)
    second = fit_conditional_logit(train_df, feature_columns, max_iter=80, random_state=random_state)
    delta = float(np.max(np.abs(first.coef_ - second.coef_))) if feature_columns else 0.0
    return _result("test_reproducibility", bool(delta <= 1e-10), max_coefficient_delta=delta, random_state=random_state)


def test_noise_feature_is_ignored(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    random_state: int = 42,
) -> HealthCheckResult:
    """Add a random feature and ensure it is not more influential than signal."""
    if train_df.empty or test_df.empty:
        return _result("test_noise_feature_is_ignored", False, reason="empty split")
    rng = np.random.default_rng(random_state)
    train = train_df.copy()
    test = test_df.copy()
    train["_random_noise"] = rng.normal(size=len(train))
    test["_random_noise"] = rng.normal(size=len(test))
    model = fit_conditional_logit(train, [*feature_columns, "_random_noise"], max_iter=80, random_state=random_state)
    signal = np.abs(model.coef_[:-1])
    noise = float(abs(model.coef_[-1]))
    threshold = float(np.quantile(signal, 0.75)) if signal.size else 0.0
    return _result(
        "test_noise_feature_is_ignored",
        bool(noise <= max(threshold, 1e-8)),
        noise_importance=noise,
        signal_75th_percentile=threshold,
    )


def test_parameter_update(artifact: BenterTwoStageArtifact) -> HealthCheckResult:
    coefficients = np.asarray(artifact.stage1_model.coef_, dtype=float)
    changed = bool(np.linalg.norm(coefficients) > 1e-8)
    return _result("test_parameter_update", changed, coefficient_norm=float(np.linalg.norm(coefficients)))


def test_learning_curve_improves(artifact: BenterTwoStageArtifact) -> HealthCheckResult:
    train_loss = list(getattr(artifact.stage1_model, "loss_history_", []))
    validation_loss = list(getattr(artifact.stage1_model, "validation_loss_history_", []))
    if len(train_loss) < 2:
        return _result("test_learning_curve_improves", False, reason="loss history unavailable")
    return _result(
        "test_learning_curve_improves",
        bool(train_loss[-1] < train_loss[0]),
        first_train_loss=float(train_loss[0]),
        last_train_loss=float(train_loss[-1]),
        overfitting_warning=bool(validation_loss and validation_loss[-1] - train_loss[-1] > 0.25),
    )


def _metrics(frame: pd.DataFrame, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "log_loss": _log_loss(frame, probabilities),
        "top1": _race_top1(frame, probabilities),
    }


def _calibration_curve(frame: pd.DataFrame, probabilities: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    if frame.empty:
        return []
    rows: list[dict[str, float]] = []
    labels = frame["is_winner"].to_numpy(dtype=float)
    for low, high in zip(np.linspace(0.0, 1.0, bins + 1)[:-1], np.linspace(0.0, 1.0, bins + 1)[1:]):
        mask = (probabilities >= low) & (probabilities <= high if high == 1.0 else probabilities < high)
        if np.any(mask):
            rows.append({"bin_low": float(low), "bin_high": float(high), "predicted": float(np.mean(probabilities[mask])), "observed": float(np.mean(labels[mask]))})
    return rows


def feature_importance_report(artifact: BenterTwoStageArtifact) -> list[dict[str, float | str]]:
    coefficients = np.asarray(artifact.stage1_model.coef_, dtype=float)
    rows = [
        {"feature": name, "importance": float(abs(coefficients[index])), "coefficient": float(coefficients[index])}
        for index, name in enumerate(artifact.feature_columns)
        if index < len(coefficients)
    ]
    return sorted(rows, key=lambda row: float(row["importance"]), reverse=True)


def _learning_curve_report(artifact: BenterTwoStageArtifact) -> dict[str, Any]:
    train_loss = list(getattr(artifact.stage1_model, "loss_history_", []))
    validation_loss = list(getattr(artifact.stage1_model, "validation_loss_history_", []))
    first = train_loss[0] if train_loss else None
    last = train_loss[-1] if train_loss else None
    return {
        "train_loss": [float(x) for x in train_loss],
        "validation_loss": [float(x) for x in validation_loss],
        "loss_decreased": bool(first is not None and last is not None and last < first),
        "overfitting_warning": bool(train_loss and validation_loss and validation_loss[-1] - train_loss[-1] > 0.25),
    }


def dataset_quality_report(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    frame = pd.concat([train_df, validation_df, test_df], ignore_index=True, sort=False)
    dedupe_columns = [
        column
        for column in ("date", "race_id", "horse_id", "draw")
        if column in frame.columns
    ]
    if dedupe_columns:
        frame = frame.drop_duplicates(subset=dedupe_columns, keep="first")
    dates = pd.to_datetime(frame.get("date", pd.Series(dtype="object")), errors="coerce").dropna()
    races = frame.get("race_id", pd.Series(dtype="object")).nunique()
    return {
        "rows": int(len(frame)),
        "unique_dates": int(dates.dt.date.nunique()),
        "unique_races": int(races),
        "winner_labels": int(pd.to_numeric(frame.get("is_winner", pd.Series(dtype=float)), errors="coerce").sum()),
        "calibration_rows": int(len(validation_df)),
        "production_ready": bool(
            dates.dt.date.nunique() >= 30
            and races >= 100
            and len(frame) >= 1000
            and len(validation_df) >= 300
        ),
        "warnings": [
            warning
            for condition, warning in [
                (dates.dt.date.nunique() < 30, "En az 30 farkli tarih gerekli."),
                (races < 100, "En az 100 yaris gerekli."),
                (len(frame) < 1000, "En az 1000 at-yaris satiri gerekli."),
                (len(validation_df) < 300, "Kalibrasyon icin en az 300 satir gerekli."),
            ]
            if condition
        ],
    }


def run_model_health_checks(
    artifact: BenterTwoStageArtifact,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    checks = [
        test_no_leakage(train_df, validation_df, test_df, feature_columns),
        test_test_set_isolation(train_df, validation_df, test_df, feature_columns),
        test_beats_baseline(artifact, train_df, test_df, feature_columns),
        test_label_shuffle_fails(train_df, test_df, feature_columns),
        test_prediction_diversity(artifact, test_df),
        test_reproducibility(train_df, test_df, feature_columns),
        test_noise_feature_is_ignored(train_df, test_df, feature_columns),
        test_parameter_update(artifact),
        test_learning_curve_improves(artifact),
    ]
    train_probability = predict_two_stage_probability(artifact, train_df)
    validation_probability = predict_two_stage_probability(artifact, validation_df)
    test_probability = predict_two_stage_probability(artifact, test_df)
    report: dict[str, Any] = {
        "passed": all(check.passed for check in checks),
        "checks": [asdict(check) for check in checks],
        "learning_curve": _learning_curve_report(artifact),
        "feature_importance": feature_importance_report(artifact),
        "feature_availability": feature_availability_report(feature_columns),
        "metrics": {
            "train": _metrics(train_df, train_probability),
            "validation": _metrics(validation_df, validation_probability),
            "test": _metrics(test_df, test_probability),
        },
        "calibration_curve": _calibration_curve(test_df, test_probability),
        "test_probability_variance": float(np.var(test_probability)) if not test_df.empty else 0.0,
        "test_rows": int(len(test_df)),
        "dataset_quality": dataset_quality_report(train_df, validation_df, test_df),
    }
    report["passed"] = bool(report["passed"] and report["learning_curve"]["loss_decreased"])
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=True, indent=2, default=str), encoding="utf-8")
    return report
