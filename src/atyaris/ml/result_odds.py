"""Gunluk sonuc CSV'sinden kapanis ganyani, AGF ve 6'li ganyan ikramiyesi toplama.

Tek istek = bir tarih/hipodrom; mevcut TJK istemcisinin hiz siniri kullanilir.
Cikti: ``tjk_result_odds.jsonl`` (kind=horse|altili), tarih|hipodrom bazinda devam edilebilir.
"""
from __future__ import annotations

import csv
import io
import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Callable

from atyaris.data_sources.tjk_scraper import _match_hippodrome

_RACE_RE = re.compile(r"^(\d+)\.\s*Ko")
_ALTILI_RE = re.compile(r"(?:(\d)\.\s*)?6'L\S*\s*GANYAN\(([\d/]+)\)\s*:\s*([\d.]+,\d+)")


def _num(text: str) -> float | None:
    try:
        return float(text.replace(".", "").replace(",", "."))
    except ValueError:
        return None


def parse_result_csv(text: str, race_date: str, hippodrome: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    race_no: int | None = None
    cols: dict[str, int] = {}
    pos = 0
    for row in csv.reader(io.StringIO(text.lstrip("\ufeff")), delimiter=";"):
        first = row[0].strip() if row else ""
        m = _RACE_RE.match(first)
        if m:
            race_no, cols, pos = int(m.group(1)), {}, 0
            continue
        if first.casefold() == "at no":
            cols = {c.strip().casefold(): i for i, c in enumerate(row)}
            continue
        if race_no is None:
            continue
        if first.startswith("GANYAN(") or "GANYAN(" in first:
            line = ";".join(row)
            for am in _ALTILI_RE.finditer(line):
                out.append({
                    "kind": "altili", "race_date": race_date, "hippodrome": hippodrome,
                    "last_race_no": race_no, "ordinal": int(am.group(1) or 1),
                    "winners": [int(x) for x in am.group(2).split("/")], "payout": _num(am.group(3)),
                })
            continue
        if cols and first.isdigit():
            pos += 1

            def cell(name: str) -> str:
                i = cols.get(name)
                return row[i].strip() if i is not None and i < len(row) else ""

            agf = re.match(r"%?([\d.]+)\((\d+)\)", cell("agf"))
            out.append({
                "kind": "horse", "race_date": race_date, "hippodrome": hippodrome, "race_no": race_no,
                "horse_no": int(first), "horse_name": cell("at i̇smi") or row[1].strip(),
                "ganyan": _num(cell("ganyan")), "agf_pct": float(agf.group(1)) if agf else None,
                "agf_rank": int(agf.group(2)) if agf else None, "csv_order": pos,
            })
    return out


def record_odds_snapshot(races: list[Any], path: Path, now: Any = None) -> int:
    """Yaris oncesi gorulen oranlari zaman damgasiyla ekler (modele girmez; sadece ileride olcum icin)."""
    from datetime import datetime

    stamp = (now or datetime.now()).isoformat(timespec="seconds")
    rows = []
    for race in races:
        for entry in getattr(race, "entries", []):
            odds = getattr(entry, "odds", None)
            if odds and odds > 1.0:
                rows.append({
                    "captured_at": stamp, "race_date": race.start_time.date().isoformat(),
                    "hippodrome": race.hippodrome, "race_no": race.race_no,
                    "horse_no": entry.number, "horse_name": entry.horse_name, "odds_prerace": float(odds),
                })
    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def collect_result_odds(
    source: Any,
    units: list[tuple[str, str]],
    out_path: Path,
    progress: Callable[[int, int, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, int]:
    """Eksik (tarih, hipodrom) birimlerini ceker; mevcutlar tekrar cekilmez."""
    done: set[str] = set()
    marker = out_path.with_suffix(".done.json")
    if marker.exists():
        done = set(json.loads(marker.read_text(encoding="utf-8")))
    todo = [u for u in units if f"{u[0]}|{u[1]}" not in done]
    stats = {"total": len(units), "fetched": 0, "empty": 0, "failed": 0, "skipped": len(units) - len(todo)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for i, (d, h) in enumerate(todo, 1):
        if should_stop and should_stop():
            break
        matched = _match_hippodrome(h) or h
        try:
            text = source._get_csv(source._daily_results_csv_url(date.fromisoformat(d), matched)) or ""
        except Exception:  # noqa: BLE001
            stats["failed"] += 1
            continue
        recs = parse_result_csv(text, d, h) if text else []
        if recs:
            with out_path.open("a", encoding="utf-8") as fh:
                for r in recs:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            stats["fetched"] += 1
        else:
            stats["empty"] += 1
        done.add(f"{d}|{h}")
        if i % 20 == 0 or i == len(todo):
            marker.write_text(json.dumps(sorted(done)), encoding="utf-8")
        if progress:
            progress(i, len(todo), f"{d} {h}")
    marker.write_text(json.dumps(sorted(done)), encoding="utf-8")
    return stats
