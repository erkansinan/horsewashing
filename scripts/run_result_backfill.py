"""Geriye donuk sonuc CSV toplama. Duraklatmak icin data/raw/tjk_result_odds.pause dosyasi olusturun."""
import json
import warnings
from datetime import date
from pathlib import Path

warnings.simplefilter("ignore")
from atyaris.config import get_settings
from atyaris.ml.cli import _tjk_source
from atyaris.ml.result_odds import collect_backfill

raw = Path("data/raw")
pf = raw / "tjk_result_odds.backfill.progress.json"
pause = raw / "tjk_result_odds.pause"
months = 24
newest = date(2026, 2, 28)
targets = {}
d = newest
total = newest.year * 12 + newest.month - 1 - months
oldest = date(total // 12, total % 12 + 1, 1)
from datetime import timedelta
while d >= oldest:
    targets[d.strftime("%Y-%m")] = targets.get(d.strftime("%Y-%m"), 0) + 1
    d -= timedelta(days=1)


def prog(stats):
    pf.write_text(json.dumps({"status": "running", "targets_days_by_month": targets, **stats}))


st = collect_backfill(_tjk_source(get_settings()), newest, months, raw / "tjk_result_odds.jsonl", prog, pause.exists)
status = "paused" if pause.exists() else ("completed" if st["days_done"] >= st["days_total"] else "partial")
pf.write_text(json.dumps({"status": status, "targets_days_by_month": targets, **st}))
print(status, st)
