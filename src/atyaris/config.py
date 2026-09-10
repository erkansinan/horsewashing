"""Uygulama ayarlari: veri, Benter pipeline, EV/Kelly ve raporlama parametreleri.

Degerler ortam degiskenleri (.env, ``ATYARIS_`` on ekiyle) veya ``config.yaml``
uzerinden gecersiz kilinabilir.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ScoringWeights(BaseModel):
    """Tahmin skorunu olusturan bilesenlerin agirliklari (toplami 1.0 olmali)."""

    form: float = 0.30
    jockey_trainer: float = 0.20
    distance_surface: float = 0.20
    weight: float = 0.15
    rest: float = 0.15

    @model_validator(mode="after")
    def _check_sum(self) -> "ScoringWeights":
        total = self.form + self.jockey_trainer + self.distance_surface + self.weight + self.rest
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Agirliklarin toplami 1.0 olmali, hesaplanan: {total:.3f}")
        return self


class Settings(BaseSettings):
    """Merkezi uygulama ayarlari.

    Not:
    - Benter tabanli aktif ML akisi `phase1_*`, `phase3_*`, `phase4_*`, `phase5_*`
      ve `ev_*` gruplarini kullanir.
    - `ensemble_*` ve `calibration_temperature` alanlari klasik prediction
      katmaninin geriye uyumlulugu icin korunur.
    """

    model_config = SettingsConfigDict(env_prefix="ATYARIS_", env_file=".env", extra="ignore")

    tjk_base_url: str = "https://www.tjk.org"
    request_timeout_seconds: float = 15.0
    min_request_interval_seconds: float = 1.0
    user_agent: str = "atyaris-tahmin/1.0 (+iletisim@example.com; arastirma/egitim amacli)"
    cache_ttl_seconds: int = 600
    cache_path: str = ".cache/atyaris_cache.sqlite3"
    recent_form_window: int = 8
    ideal_rest_days_min: int = 14
    ideal_rest_days_max: int = 45

    # Legacy klasik prediction ayarlari (Benter ML pipeline tarafinda kullanilmaz).
    ensemble_boosting_weight: float = 0.58
    ensemble_ranking_weight: float = 0.42
    calibration_temperature: float = 0.85

    ev_probability_threshold: float = 0.18
    ev_min_edge: float = 0.03
    ev_min_value: float = 0.02
    ev_fractional_kelly: float = 0.35
    ev_max_kelly_fraction: float = 0.25

    backtest_lookback_days: int = 8
    backtest_leakage_safe_mode: bool = True
    backtest_exclude_recent_races: int = 1
    backtest_optimize_calibration: bool = True

    # Benter pipeline veri/artifact path ayarlari.
    phase1_raw_csv_path: str = "data/raw/tjk_real_races.csv"
    phase1_clean_csv_path: str = "data/processed/clean_races.csv"
    phase1_features_csv_path: str = "data/processed/features_phase1.csv"
    phase1_prediction_features_csv_path: str = "data/processed/prediction_features_phase1.csv"
    phase1_model_path: str = "models/phase1_logreg.joblib"
    phase1_min_train_days: int = 90
    phase1_holdout_days: int = 30

    phase2_enable_advanced_features: bool = True
    phase2_style_front_threshold: float = 0.6
    phase2_style_presser_threshold: float = 0.2
    phase2_layoff_days: int = 75

    # Olasilik kalibrasyonu: `isotonic`, `platt`, `none`.
    phase3_calibration_method: str = "isotonic"
    phase3_calibration_days: int = 21
    phase3_enable_ev: bool = True

    # 6'li kupon optimizasyon parametreleri.
    phase4_default_budget: float = 500.0
    phase4_unit_cost: float = 1.0
    phase4_beam_width: int = 80
    phase4_top_per_leg: int = 4
    phase4_simulation_count: int = 20000
    phase4_base_payout: float = 120000.0
    phase4_no_bet_ev_threshold: float = 0.0
    phase4_min_confidence: float = 0.35
    phase4_concentration_penalty: float = 0.15

    # Raporlama ve deney izleme.
    phase5_tracking_db_path: str = "data/database/experiments.sqlite3"
    phase5_report_dir: str = "reports"
    phase5_top_feature_count: int = 15
    phase5_auto_promote_candidate: bool = True

    weights: ScoringWeights = ScoringWeights()

    @classmethod
    def from_yaml(cls, path: str | Path = "config.yaml") -> "Settings":
        yaml_path = Path(path)
        data: dict = {}
        if yaml_path.exists():
            with yaml_path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        return cls(**data)


def get_settings() -> Settings:
    """Uygulama genelinde kullanilacak ayarlari yukler (config.yaml + .env)."""
    return Settings.from_yaml()
