from __future__ import annotations

from datetime import date, datetime

from atyaris.data_sources.tjk_scraper import TJKHtmlDataSource
from atyaris.ml.horse_id_mapping import HorseIdMappingStore, resolve_race_horse_ids
from atyaris.models.entities import Jockey, Race, RaceEntry, TrackSurface, Trainer


def _race_entry(name: str) -> RaceEntry:
    return RaceEntry(
        number=1,
        horse_id=f"{name}-1",
        horse_name=name,
        age=3,
        jockey=Jockey(name="Jokey"),
        trainer=Trainer(name="Antrenor"),
        weight_kg=57.0,
    )


def _race(entry: RaceEntry, target_date: date) -> Race:
    return Race(
        id=f"Ankara-{target_date.isoformat()}-1",
        hippodrome="Ankara",
        race_no=1,
        start_time=datetime.combine(target_date, datetime.min.time()),
        distance_m=1200,
        surface=TrackSurface.KUM,
        entries=[entry],
    )


def test_mapping_persists_same_name_as_separate_ids_and_reports_collision(tmp_path) -> None:
    path = tmp_path / "horse_ids.jsonl"
    mapping = HorseIdMappingStore(path)

    assert mapping.observe("TEKRAR", 101, date(2020, 1, 1)) == []
    assert mapping.observe("TEKRAR", 202, date(2025, 1, 1)) == [101, 202]

    reloaded = HorseIdMappingStore(path)
    assert len(reloaded.records()) == 2
    assert reloaded.resolve_for_date("TEKRAR", date(2020, 1, 1)) == (
        101,
        "cached",
        [101],
    )
    assert reloaded.resolve_for_date("TEKRAR", date(2025, 1, 1)) == (
        202,
        "cached",
        [202],
    )
    assert HorseIdMappingStore(tmp_path / "unique.jsonl").resolve_for_date(
        "TEKRAR", date(2020, 1, 1)
    ) == (None, "unresolved", [])
    assert reloaded.collision_report()["names_with_multiple_ids"] == 1
    assert all(item["confidence_source"] == "html_program_link" for item in reloaded.records())


def test_mapping_does_not_resolve_overlapping_name_collision(tmp_path) -> None:
    mapping = HorseIdMappingStore(tmp_path / "horse_ids.jsonl")
    mapping.observe("TEKRAR", 101, date(2025, 1, 1))
    mapping.observe("TEKRAR", 202, date(2025, 1, 1))

    assert mapping.resolve_for_date("TEKRAR", date(2025, 1, 1)) == (
        None,
        "ambiguous",
        [101, 202],
    )


def test_resolver_uses_date_validated_cache_without_html_or_search(tmp_path) -> None:
    target_date = date(2025, 1, 1)
    mapping = HorseIdMappingStore(tmp_path / "horse_ids.jsonl")
    mapping.observe("TEKRAR", 101, target_date)
    source = TJKHtmlDataSource()
    source.get_daily_races_html = lambda *_args: (_ for _ in ()).throw(  # type: ignore[method-assign]
        AssertionError("cached identity should skip program HTML")
    )
    source.get_daily_result_horse_ids_html = lambda *_args: (_ for _ in ()).throw(  # type: ignore[method-assign]
        AssertionError("cached identity should skip result HTML")
    )
    source.find_horse_id_from_search = lambda *_args: (_ for _ in ()).throw(  # type: ignore[method-assign]
        AssertionError("cached identity should skip Atlar search")
    )
    races = [_race(_race_entry("TEKRAR"), target_date)]

    report = resolve_race_horse_ids(source, races, target_date, "Ankara", mapping)

    assert races[0].entries[0].source_horse_id == 101
    assert races[0].entries[0].id_resolution_status == "cached"
    assert races[0].entries[0].id_unresolved is False
    assert report["resolved"] == 1
    assert report["unresolved"] == 0


def test_validate_random_mappings_reports_name_checks_without_network(tmp_path) -> None:
    from atyaris.ml.horse_id_mapping import validate_random_mappings

    mapping = HorseIdMappingStore(tmp_path / "horse_ids.jsonl")
    mapping.observe("AT A", 101, date(2025, 1, 1))
    mapping.observe("AT B", 202, date(2025, 1, 1))
    source = TJKHtmlDataSource()
    source.validate_horse_id_name = lambda horse_id, _name: (  # type: ignore[method-assign]
        "matched" if horse_id == 101 else "mismatch",
        ["AT A" if horse_id == 101 else "FARKLI"],
    )

    report = validate_random_mappings(source, mapping, sample_size=20, seed=1)

    assert report["sample_size_checked"] == 2
    assert report["matched"] == 1
    assert report["mismatched"] == 1