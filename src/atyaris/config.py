"""Uygulama ayarlari: skorlama agirliklari, HTTP ve cache parametreleri.

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
    """Merkezi uygulama ayarlari."""

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
    ensemble_boosting_weight: float = 0.58
    ensemble_ranking_weight: float = 0.42
    calibration_temperature: float = 0.85
    ev_probability_threshold: float = 0.18
    ev_min_edge: float = 0.03
    ev_min_value: float = 0.02
    ev_fractional_kelly: float = 0.35
    ev_max_kelly_fraction: float = 0.25
    backtest_lookback_days: int = 8
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
