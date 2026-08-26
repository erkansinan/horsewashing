from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Protocol

import numpy as np
import pandas as pd


class RacingDataProvider(Protocol):
    """Provider interface for race-centric ingestion."""

    def get_races(self, start_date: date, end_date: date) -> pd.DataFrame:
        ...

    def get_entries(self, start_date: date, end_date: date) -> pd.DataFrame:
        ...

    def get_results(self, start_date: date, end_date: date) -> pd.DataFrame:
        ...

    def get_odds(self, start_date: date, end_date: date) -> pd.DataFrame:
        ...


@dataclass
class SyntheticProviderConfig:
    races_per_day: int = 6
    min_field_size: int = 7
    max_field_size: int = 12
    horse_pool_size: int = 260
    random_state: int = 42


class SyntheticRacingDataProvider:
    """Synthetic provider for Phase 1 pipeline validation.

    This generator is intentionally simplified and is used only to validate the
    data/feature/model/backtest flow under leakage-safe constraints.
    """

    def __init__(self, config: SyntheticProviderConfig | None = None) -> None:
        self.config = config or SyntheticProviderConfig()

    def _base_frames(self, start_date: date, end_date: date) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(self.config.random_state)
        days = pd.date_range(start=start_date, end=end_date, freq="D")

        races: list[dict] = []
        entries: list[dict] = []

        horse_base_skill = rng.normal(loc=0.0, scale=1.0, size=self.config.horse_pool_size)
        horse_names = [f"HORSE_{i:04d}" for i in range(self.config.horse_pool_size)]

        for d in days:
            for race_no in range(1, self.config.races_per_day + 1):
                race_id = f"{d.strftime('%Y%m%d')}_{race_no:02d}"
                field_size = int(rng.integers(self.config.min_field_size, self.config.max_field_size + 1))
                track = str(rng.choice(["ANKARA", "ISTANBUL", "IZMIR"]))
                surface = str(rng.choice(["KUM", "CIM", "SENTETIK"], p=[0.4, 0.4, 0.2]))
                distance = int(rng.choice([1200, 1400, 1600, 1800, 2000]))
                temperature = float(rng.normal(20.0, 7.0))
                wind = float(abs(rng.normal(8.0, 3.0)))
                start_dt = datetime(d.year, d.month, d.day, 12, 0) + timedelta(minutes=35 * race_no)

                races.append(
                    {
                        "race_id": race_id,
                        "date": d.date(),
                        "race_datetime": start_dt,
                        "track": track,
                        "country": "TR",
                        "surface": surface,
                        "track_condition": str(rng.choice(["NORMAL", "NEMLI", "AGIR"])),
                        "distance": distance,
                        "race_class": str(rng.choice(["MAIDEN", "HANDIKAP", "SARTLI", "KV"])),
                        "race_type": str(rng.choice(["A", "B", "C"])),
                        "weather": str(rng.choice(["SUNNY", "CLOUDY", "RAIN"])),
                        "temperature": temperature,
                        "wind": wind,
                        "field_size": field_size,
                    }
                )

                horse_ids = rng.choice(np.arange(self.config.horse_pool_size), size=field_size, replace=False)
                base_pace = rng.normal(0.0, 1.0, size=field_size)
                fatigue = rng.normal(0.0, 1.0, size=field_size)
                draw_noise = rng.normal(0.0, 0.15, size=field_size)
                score = horse_base_skill[horse_ids] + 0.25 * base_pace - 0.2 * fatigue + draw_noise
                # Softmax-ish latent winner probabilities for synthetic truth.
                exp_s = np.exp(score - np.max(score))
                win_probs = exp_s / exp_s.sum()
                winner_idx = int(rng.choice(np.arange(field_size), p=win_probs))

                ranking = np.argsort(-score)
                finish_positions = np.empty(field_size, dtype=int)
                finish_positions[ranking] = np.arange(1, field_size + 1)
                finish_positions[winner_idx] = 1

                for i, horse_ix in enumerate(horse_ids):
                    true_p = float(win_probs[i])
                    margin = float(rng.normal(0.0, 0.07))
                    market_p = min(max(true_p + margin, 0.01), 0.85)
                    odds = round(max(1.01, 1.0 / market_p), 2)

                    entries.append(
                        {
                            "race_id": race_id,
                            "date": d.date(),
                            "race_datetime": start_dt,
                            "horse_id": f"H{int(horse_ix):04d}",
                            "horse_name": horse_names[int(horse_ix)],
                            "age": int(rng.integers(3, 8)),
                            "sex": str(rng.choice(["M", "F"])),
                            "weight": float(round(rng.normal(56.0, 2.2), 1)),
                            "draw": i + 1,
                            "jockey": f"J_{int(rng.integers(1, 80)):03d}",
                            "trainer": f"T_{int(rng.integers(1, 60)):03d}",
                            "equipment": str(rng.choice(["NONE", "KG", "DB", "SK"])),
                            "odds": odds,
                            "market_probability": market_p,
                            "latent_true_win_probability": true_p,
                            "finish_position": int(finish_positions[i]),
                            "is_winner": int(finish_positions[i] == 1),
                            "early_pace": float(base_pace[i]),
                            "fatigue_signal": float(fatigue[i]),
                        }
                    )

        races_df = pd.DataFrame(races)
        entries_df = pd.DataFrame(entries)
        return races_df, entries_df

    def get_races(self, start_date: date, end_date: date) -> pd.DataFrame:
        races_df, _ = self._base_frames(start_date, end_date)
        return races_df

    def get_entries(self, start_date: date, end_date: date) -> pd.DataFrame:
        _, entries_df = self._base_frames(start_date, end_date)
        return entries_df.drop(columns=["finish_position", "is_winner"], errors="ignore")

    def get_results(self, start_date: date, end_date: date) -> pd.DataFrame:
        _, entries_df = self._base_frames(start_date, end_date)
        return entries_df[["race_id", "horse_id", "finish_position", "is_winner"]].copy()

    def get_odds(self, start_date: date, end_date: date) -> pd.DataFrame:
        _, entries_df = self._base_frames(start_date, end_date)
        return entries_df[["race_id", "horse_id", "odds", "market_probability"]].copy()

    def get_dataset(self, start_date: date, end_date: date) -> pd.DataFrame:
        races_df, entries_df = self._base_frames(start_date, end_date)
        return entries_df.merge(races_df, on=["race_id", "date", "race_datetime"], how="left")
