"""Veri toplama isi icin gercekci gun ilerlemesi ve sure tahmini.

Tahmin, gun sayisini degil gunlerin gercek maliyetini kullanir: checkpoint'ten
zaten tamamlanmis gunler (ag istegi gerektirmeyen) ile gercekten veri cekilen
gunler ayri hesaplanir ve son gunlerin medyani baz alinir.
"""
from __future__ import annotations

from collections import deque
from datetime import date, datetime, timedelta
from statistics import median
from time import monotonic
from typing import Callable, Mapping, Sequence

COMPLETE = "complete"
NO_RACE = "no_race"
WARNING = "warning"
SKIPPED = "skipped"

# Tahmin edilen gun tipleri: ag gerektirmez / yalnizca CSV kontrolu / tam veri cekimi.
ZERO = "zero"
LIGHT = "light"
WORKED = "worked"

_COST_WINDOW = 5


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return "1 dk'dan az"
    minutes = int(round(seconds / 60.0))
    if minutes < 60:
        return f"{minutes} dk"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} sa {minutes} dk" if minutes else f"{hours} sa"


class CollectionProgress:
    def __init__(
        self,
        days: Sequence[date],
        predicted_types: Mapping[date, str],
        hippodrome_count: int,
        light_day_seconds: float = 15.0,
        clock: Callable[[], float] = monotonic,
        now: Callable[[], datetime] = datetime.now,
    ) -> None:
        self._days = list(days)
        self._predicted = dict(predicted_types)
        self._hippodrome_count = max(1, hippodrome_count)
        self._light_default = max(0.0, light_day_seconds)
        self._clock = clock
        self._now = now
        self._statuses: dict[date, str] = {}
        self._costs: dict[str, deque[float]] = {
            WORKED: deque(maxlen=_COST_WINDOW),
            LIGHT: deque(maxlen=_COST_WINDOW),
        }
        self._active: date | None = None
        self._active_started = 0.0
        self._active_worked = False
        self._active_network = False
        self._hippodromes_done = 0

    def start_day(self, day: date) -> None:
        self._active = day
        self._active_started = self._clock()
        self._active_worked = False
        self._active_network = False
        self._hippodromes_done = 0

    def hippodromes_done(self, done: int) -> None:
        self._hippodromes_done = max(0, min(done, self._hippodrome_count))

    def note_network(self) -> None:
        self._active_network = True

    def mark_worked(self) -> None:
        self._active_network = True
        self._active_worked = True

    def finish_day(self, status: str) -> None:
        if self._active is None:
            return
        cost = max(self._clock() - self._active_started, 0.0)
        if self._active_worked:
            self._costs[WORKED].append(cost)
        elif self._active_network:
            self._costs[LIGHT].append(cost)
        self._statuses[self._active] = status
        self._active = None
        self._hippodromes_done = 0

    def _estimates(self) -> tuple[float | None, float, str]:
        worked = list(self._costs[WORKED])
        light = list(self._costs[LIGHT])
        active_fraction = self._hippodromes_done / self._hippodrome_count
        provisional = False
        est_worked: float | None = median(worked) if worked else None
        if est_worked is None and self._active is not None and self._active_worked and active_fraction >= 0.2:
            est_worked = (self._clock() - self._active_started) / active_fraction
            provisional = True
        est_light = median(light) if light else self._light_default
        if est_worked is None:
            confidence = "olculuyor"
        elif provisional or len(worked) < 3:
            confidence = "orta"
        else:
            confidence = "yuksek"
        return est_worked, est_light, confidence

    def snapshot(self) -> dict[str, object]:
        total = max(len(self._days), 1)
        counts = {COMPLETE: 0, NO_RACE: 0, WARNING: 0, SKIPPED: 0}
        for status in self._statuses.values():
            counts[status] = counts.get(status, 0) + 1
        processed = len(self._statuses)
        active_fraction = self._hippodromes_done / self._hippodrome_count if self._active else 0.0
        percent = min((processed + active_fraction) / total * 100.0, 100.0)
        if self._active is not None:
            percent = min(percent, 99.9)

        est_worked, est_light, confidence = self._estimates()
        remaining_seconds = 0.0
        unknown_days = 0
        if self._active is not None:
            elapsed = max(self._clock() - self._active_started, 0.0)
            kind = WORKED if self._active_worked else self._predicted.get(self._active, WORKED)
            if kind == ZERO:
                expected = 0.0
            elif kind == LIGHT:
                expected = est_light
            elif est_worked is not None:
                expected = est_worked
            else:
                expected = None
            if expected is None:
                unknown_days += 1
            elif elapsed >= expected and active_fraction >= 0.05:
                remaining_seconds += elapsed * (1.0 - active_fraction) / active_fraction
            else:
                remaining_seconds += max(expected - elapsed, 0.0)
        for day in self._days:
            if day in self._statuses or day == self._active:
                continue
            kind = self._predicted.get(day, WORKED)
            if kind == ZERO:
                continue
            if kind == LIGHT:
                remaining_seconds += est_light
            elif est_worked is not None:
                remaining_seconds += est_worked
            else:
                unknown_days += 1

        pending_days = total - processed
        if pending_days <= 0:
            eta_seconds: float | None = 0.0
            text = "tamamlandi"
            finish_at = None
        elif unknown_days:
            eta_seconds = None
            text = "olculuyor; ilk veri cekilen gun bitince hesaplanir"
            finish_at = None
        else:
            eta_seconds = remaining_seconds
            text = f"~{format_duration(remaining_seconds)}"
            finish_at = (self._now() + timedelta(seconds=remaining_seconds)).strftime("%H:%M")
            if confidence != "yuksek":
                text += " (yaklasik)"
        return {
            "processed_days": processed,
            "completed_days": counts[COMPLETE] + counts[NO_RACE],
            "warning_days": counts[WARNING],
            "remaining_days": max(pending_days, 0),
            "progress_percent": round(percent, 1),
            "estimated_remaining": text,
            "eta_seconds": eta_seconds,
            "estimated_finish_at": finish_at,
            "eta_confidence": confidence,
            "day_statuses": {day.isoformat(): status for day, status in self._statuses.items()},
        }
