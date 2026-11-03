from atyaris.ml.coupon import allocate, fixed_alloc
from atyaris.ml.local_pipeline import _horse_name_key
from atyaris.ml.result_odds import parse_result_csv

_CSV = (
    "Ankara;(1. Gun);15/09/2026\n\n"
    "1. Kosu :   14.00;Maiden;;;;;;;;Rekor\n\n"
    "At No;At İsmi;Yas;B;A;Kilo;Jokey;Sahip;Antrenor;St;AGF;H;Derece;Ganyan;Fark\n"
    "1;KING ULUDAG KG DB;3y;x;y;59;J;S;T;9;%17.57(2);35;1:31.24;3,10;Bas\n"
    "2;ARKANBERK KG;3y;x;y;59;J;S;T;5;%12.91(3);36;1:31.27;13,85;Burun\n"
    "GANYAN(1) :3,10 TL, 1. 6'LI GANYAN(1/2/3/4/5/6) :1.253,37 TL\n"
)


def test_parse_result_csv_reads_odds_agf_and_altili() -> None:
    rows = parse_result_csv(_CSV, "2026-09-15", "Ankara")
    horses = [r for r in rows if r["kind"] == "horse"]
    assert [h["ganyan"] for h in horses] == [3.10, 13.85]
    assert horses[0]["agf_pct"] == 17.57 and horses[0]["agf_rank"] == 2
    altili = [r for r in rows if r["kind"] == "altili"]
    assert altili[0]["winners"] == [1, 2, 3, 4, 5, 6] and altili[0]["payout"] == 1253.37


def test_horse_name_key_ignores_equipment_suffixes_and_notes() -> None:
    assert _horse_name_key("BATMAN KIZI KG K") == _horse_name_key("BATMAN KIZI")
    assert _horse_name_key("BUZMERAL  K Kulaklık takılacağını") == _horse_name_key("BUZMERAL K")


def test_allocate_respects_budget_and_prefers_uncertain_legs() -> None:
    import numpy as np

    legs = [np.array([0.9, 0.05, 0.05])] + [np.array([0.3, 0.3, 0.2, 0.2])] * 5
    ks = allocate(legs, 48)
    assert int(np.prod(ks)) <= 48 and ks[0] == 1
    assert int(np.prod(fixed_alloc(legs, 48))) <= 48
