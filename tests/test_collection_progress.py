from datetime import date, datetime

from atyaris.web.collection_progress import (
    COMPLETE,
    LIGHT,
    NO_RACE,
    SKIPPED,
    WARNING,
    WORKED,
    ZERO,
    CollectionProgress,
    format_duration,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


DAYS = [date(2026, 10, 30 - offset) for offset in range(6)]
NOW = datetime(2026, 11, 1, 20, 0, 0)


def _tracker(types: dict[date, str], clock: FakeClock) -> CollectionProgress:
    return CollectionProgress(
        DAYS, types, hippodrome_count=10, light_day_seconds=10.0, clock=clock, now=lambda: NOW
    )


def test_format_duration() -> None:
    assert format_duration(20) == "1 dk'dan az"
    assert format_duration(45 * 60) == "45 dk"
    assert format_duration(2 * 3600) == "2 sa"
    assert format_duration(2 * 3600 + 15 * 60) == "2 sa 15 dk"


def test_eta_is_unknown_until_first_worked_day_has_data() -> None:
    clock = FakeClock()
    tracker = _tracker({day: WORKED for day in DAYS}, clock)
    tracker.start_day(DAYS[0])

    snapshot = tracker.snapshot()

    assert snapshot["eta_seconds"] is None
    assert snapshot["eta_confidence"] == "olculuyor"
    assert snapshot["remaining_days"] == 6
    assert snapshot["processed_days"] == 0


def test_eta_uses_median_of_worked_days_and_counts_warning_days_as_processed() -> None:
    clock = FakeClock()
    tracker = _tracker({day: WORKED for day in DAYS}, clock)
    for index, status in enumerate([WARNING, COMPLETE, COMPLETE]):
        tracker.start_day(DAYS[index])
        tracker.mark_worked()
        clock.advance(600.0)
        tracker.finish_day(status)

    snapshot = tracker.snapshot()

    assert snapshot["processed_days"] == 3
    assert snapshot["completed_days"] == 2
    assert snapshot["warning_days"] == 1
    assert snapshot["remaining_days"] == 3
    assert snapshot["eta_seconds"] == 3 * 600.0
    assert snapshot["estimated_remaining"] == "~30 dk"
    assert snapshot["estimated_finish_at"] == "20:30"
    assert snapshot["eta_confidence"] == "yuksek"
    assert snapshot["progress_percent"] == 50.0
    assert snapshot["day_statuses"]["2026-10-30"] == WARNING


def test_zero_cost_days_do_not_inflate_the_estimate() -> None:
    clock = FakeClock()
    types = {day: WORKED for day in DAYS}
    types.update({DAYS[3]: ZERO, DAYS[4]: ZERO, DAYS[5]: LIGHT})
    tracker = _tracker(types, clock)
    tracker.start_day(DAYS[0])
    tracker.mark_worked()
    clock.advance(1200.0)
    tracker.finish_day(COMPLETE)

    snapshot = tracker.snapshot()

    # 2 worked gun x 1200 sn + 1 hafif gun x 10 sn; sifir maliyetli gunler sayilmaz.
    assert snapshot["eta_seconds"] == 2 * 1200.0 + 10.0
    assert snapshot["remaining_days"] == 5


def test_first_day_gets_provisional_estimate_from_hippodrome_progress() -> None:
    clock = FakeClock()
    tracker = _tracker({day: WORKED for day in DAYS}, clock)
    tracker.start_day(DAYS[0])
    tracker.mark_worked()
    clock.advance(300.0)
    tracker.hippodromes_done(5)

    snapshot = tracker.snapshot()

    assert snapshot["eta_confidence"] == "orta"
    assert snapshot["eta_seconds"] == 300.0 + 5 * 600.0
    assert snapshot["estimated_remaining"].endswith("(yaklasik)")
    assert 0 < snapshot["progress_percent"] < 100


def test_skipped_and_no_race_days_are_processed_with_correct_completion_counts() -> None:
    clock = FakeClock()
    tracker = _tracker({day: ZERO for day in DAYS}, clock)
    for day, status in zip(DAYS, [SKIPPED, NO_RACE, COMPLETE, COMPLETE, COMPLETE, COMPLETE]):
        tracker.start_day(day)
        tracker.finish_day(status)

    snapshot = tracker.snapshot()

    assert snapshot["processed_days"] == 6
    assert snapshot["completed_days"] == 5
    assert snapshot["remaining_days"] == 0
    assert snapshot["estimated_remaining"] == "tamamlandi"
    assert snapshot["progress_percent"] == 100.0
