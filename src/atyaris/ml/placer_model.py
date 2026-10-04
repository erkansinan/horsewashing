from __future__ import annotations

import numpy as np
import pandas as pd

from atyaris.ml.features import TJK_HIGHEST_ODDS_PLACER_FEATURE_COLUMNS
from atyaris.ml.fundamental_model import (
    ConditionalLogitModel,
    fit_conditional_logit,
    predict_conditional_logit_probability,
)


TARGET_COLUMN = "target_highest_odds_placer"
LABEL_COLUMN = "is_highest_odds_placer"


def _labeled_placer_races(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"race_id", "horse_id", "odds", TARGET_COLUMN}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"En yuksek oranli plase modeli icin eksik kolonlar: {sorted(missing)}")

    all_labeled = frame.loc[frame[TARGET_COLUMN].notna()].copy()
    if all_labeled.empty:
        raise ValueError("En yuksek oranli plase modeli icin etiketli yaris bulunamadi")

    labeled = all_labeled.copy()
    labeled["odds"] = pd.to_numeric(labeled["odds"], errors="coerce")
    labeled = labeled.loc[labeled["odds"] > 1.0].copy()
    target_races = set(all_labeled["race_id"].astype(str))
    eligible_races = set(labeled["race_id"].astype(str))
    missing_target_races = target_races.difference(eligible_races)
    if missing_target_races:
        raise ValueError(
            "Etiketli yaris hedefi ganyani 1.0'dan buyuk bir at olmali; "
            f"hatali yarislar: {sorted(missing_target_races)[:5]}"
        )
    labeled[LABEL_COLUMN] = (
        labeled["horse_id"].astype(str)
        == labeled[TARGET_COLUMN].astype(str)
    ).astype(float)
    matches_per_race = labeled.groupby("race_id")[LABEL_COLUMN].sum()
    invalid_races = matches_per_race[matches_per_race != 1.0]
    if not invalid_races.empty:
        raise ValueError(
            "Her etiketli yarista tam bir hedef at olmali; hatali yarislar: "
            f"{invalid_races.index.astype(str).tolist()[:5]}"
        )
    return labeled


def fit_highest_odds_placer_model(frame: pd.DataFrame) -> ConditionalLogitModel:
    labeled = _labeled_placer_races(frame)
    missing_features = set(TJK_HIGHEST_ODDS_PLACER_FEATURE_COLUMNS).difference(labeled.columns)
    if missing_features:
        raise ValueError(
            "En yuksek oranli plase modeli icin eksik feature'lar: "
            f"{sorted(missing_features)}; feature'lari yeniden uretin."
        )
    invalid_columns = [
        column
        for column in TJK_HIGHEST_ODDS_PLACER_FEATURE_COLUMNS
        if not np.isfinite(labeled[column].to_numpy(dtype=float)).all()
    ]
    if invalid_columns:
        raise ValueError(
            "En yuksek oranli plase feature'lari sonlu sayisal degerlerden olusmali: "
            f"{invalid_columns}"
        )

    return fit_conditional_logit(
        labeled,
        TJK_HIGHEST_ODDS_PLACER_FEATURE_COLUMNS,
        learning_rate=0.05,
        max_iter=800,
        regularization_strength=0.5,
        label_col=LABEL_COLUMN,
    )


def predict_highest_odds_placer_probability(
    model: ConditionalLogitModel,
    frame: pd.DataFrame,
) -> np.ndarray:
    missing_features = set(model.feature_columns).difference(frame.columns)
    if missing_features:
        raise ValueError(
            "Tahmin feature'lari eksik: "
            f"{sorted(missing_features)}; feature'lari yeniden uretin."
        )
    if "odds" not in frame.columns:
        raise ValueError("En yuksek oranli plase tahmini icin ganyan kolonu gerekli")
    odds = pd.to_numeric(frame["odds"], errors="coerce")
    eligible_positions = np.flatnonzero((odds > 1.0).to_numpy())
    probabilities = np.zeros(len(frame), dtype=float)
    if eligible_positions.size == 0:
        return probabilities

    eligible_frame = frame.iloc[eligible_positions]
    invalid_columns = [
        column
        for column in model.feature_columns
        if not np.isfinite(eligible_frame[column].to_numpy(dtype=float)).all()
    ]
    if invalid_columns:
        raise ValueError(
            "En yuksek oranli plase tahmin feature'lari sonlu sayisal degerlerden "
            f"olusmali: {invalid_columns}"
        )
    probabilities[eligible_positions] = predict_conditional_logit_probability(
        model,
        eligible_frame,
    )
    return probabilities
