from __future__ import annotations

from datetime import date, timedelta
import json

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from atyaris.config import get_settings
from atyaris.data_sources.tjk_scraper import TJKHtmlDataSource
from atyaris.ml.feedback import compare_predictions
from atyaris.ml.phase5 import register_training_run, run_phase5_report
from atyaris.ml.pipeline import (
    build_features,
    ingest_real_data,
    optimize_for_date,
    paths_from_settings,
    predict_for_date,
    preprocess_raw,
    run_phase1_backtest,
    train_phase1_model,
)

app = typer.Typer(add_completion=False, help="Phase 5 ML pipeline commands")
console = Console()


def _tjk_source(settings: object) -> TJKHtmlDataSource:
    return TJKHtmlDataSource(
        base_url=settings.tjk_base_url,  # type: ignore[attr-defined]
        user_agent=settings.user_agent,  # type: ignore[attr-defined]
        request_timeout=settings.request_timeout_seconds,  # type: ignore[attr-defined]
        min_request_interval=settings.min_request_interval_seconds,  # type: ignore[attr-defined]
        cache_ttl_seconds=settings.cache_ttl_seconds,  # type: ignore[attr-defined]
    )


@app.command("ingest")
def ingest_cmd(
    start_date: str = typer.Option("2024-01-01", "--start-date"),
    end_date: str = typer.Option("2025-12-31", "--end-date"),
) -> None:
    settings = get_settings()
    paths = paths_from_settings(settings)
    data = ingest_real_data(
        date.fromisoformat(start_date),
        date.fromisoformat(end_date),
        paths,
        progress_callback=console.print,
    )
    console.print(f"Ingest tamam: {len(data)} satir -> {paths.raw_csv}")


@app.command("learn")
def learn_cmd(
    start_date: str = typer.Option(..., "--start-date", help="YYYY-MM-DD"),
    end_date: str = typer.Option(..., "--end-date", help="YYYY-MM-DD"),
    holdout_days: int = typer.Option(None, "--holdout-days"),
) -> None:
    """Gercek TJK sonuclarini cekip modeli yeniden egitir ve backtest yapar."""
    settings = get_settings()
    paths = paths_from_settings(settings)
    console.print("[bold]1/5 Veri cekme basladi[/bold]")
    data = ingest_real_data(
        date.fromisoformat(start_date),
        date.fromisoformat(end_date),
        paths,
        progress_callback=console.print,
    )
    console.print(f"[green]1/5 tamamlandi:[/green] {len(data)} satir -> {paths.raw_csv}")
    console.print("[bold]2/5 Preprocess basladi[/bold]")
    preprocess_raw(paths)
    console.print(f"[green]2/5 tamamlandi:[/green] {paths.clean_csv}")
    console.print("[bold]3/5 Feature uretimi basladi[/bold]")
    built = build_features(paths)
    console.print(f"[green]3/5 tamamlandi:[/green] {len(built.frame)} satir -> {paths.features_csv}")
    chosen_holdout = holdout_days if holdout_days is not None else settings.phase1_holdout_days
    console.print("[bold]4/5 Model egitimi basladi[/bold]")
    artifact = train_phase1_model(
        paths,
        holdout_days=chosen_holdout,
        calibration_days=settings.phase3_calibration_days,
        calibration_method=settings.phase3_calibration_method,
    )
    model_version = register_training_run(
        settings,
        paths,
        holdout_days=chosen_holdout,
        calibration_method=settings.phase3_calibration_method,
        blend_weight=float(artifact.logistic_weight),
        feature_frame=built.frame,
    )
    console.print(f"[green]4/5 tamamlandi:[/green] model={model_version}")
    console.print("[bold]5/5 Walk-forward backtest basladi[/bold]")
    backtest = run_phase1_backtest(
        paths,
        min_train_days=settings.phase1_min_train_days,
        calibration_days=settings.phase3_calibration_days,
        calibration_method=settings.phase3_calibration_method,
        ev_probability_threshold=settings.ev_probability_threshold,
        ev_min_edge=settings.ev_min_edge,
        ev_min_value=settings.ev_min_value,
    )
    console.print("[green]5/5 tamamlandi[/green]")
    console.print_json(json.dumps({"model_version": model_version, "backtest": backtest}, ensure_ascii=True))


@app.command("results")
def results_cmd(
    target_date: str = typer.Option(..., "--date", help="YYYY-MM-DD"),
    city: str = typer.Option("İstanbul", "--city", help="Hipodrom"),
    compare: bool = typer.Option(True, "--compare/--no-compare"),
) -> None:
    """TJK gunluk sonuc ozetini ceker ve varsa ML tahminleriyle karsilastirir."""
    settings = get_settings()
    paths = paths_from_settings(settings)
    dt = date.fromisoformat(target_date)
    source = _tjk_source(settings)
    try:
        results = source.get_daily_race_results(dt, city)
    finally:
        source.close()

    table = Table(title=f"TJK Sonuclari - {city} - {dt.isoformat()}")
    table.add_column("Kosu")
    table.add_column("At sayisi")
    table.add_column("Kazanan No")
    for race_no, race_results in sorted(results.items()):
        winners = [str(number) for number, position in race_results.items() if position == 1]
        table.add_row(str(race_no), str(len(race_results)), ", ".join(winners) or "-")
    console.print(table)

    if not compare or not paths.features_csv.exists() or not paths.model_path.exists():
        return
    predictions = predict_for_date(
        paths,
        dt,
        enable_ev=True,
        ev_probability_threshold=settings.ev_probability_threshold,
        ev_min_edge=settings.ev_min_edge,
        ev_min_value=settings.ev_min_value,
    )
    comparison = compare_predictions(predictions, results)
    console.print_json(
        json.dumps(
            {key: value for key, value in comparison.items() if key != "rows"},
            ensure_ascii=True,
        )
    )


@app.command("preprocess")
def preprocess_cmd() -> None:
    settings = get_settings()
    paths = paths_from_settings(settings)
    cleaned = preprocess_raw(paths)
    console.print(f"Preprocess tamam: {len(cleaned)} satir -> {paths.clean_csv}")


@app.command("features")
def features_cmd() -> None:
    settings = get_settings()
    paths = paths_from_settings(settings)
    built = build_features(paths)
    console.print(f"Feature tamam: {len(built.frame)} satir, {len(built.feature_columns)} feature -> {paths.features_csv}")


@app.command("train")
def train_cmd(
    holdout_days: int = typer.Option(None, "--holdout-days"),
) -> None:
    settings = get_settings()
    paths = paths_from_settings(settings)
    chosen_holdout = holdout_days if holdout_days is not None else settings.phase1_holdout_days
    artifact = train_phase1_model(
        paths,
        holdout_days=chosen_holdout,
        calibration_days=settings.phase3_calibration_days,
        calibration_method=settings.phase3_calibration_method,
    )
    feat = pd.read_csv(paths.features_csv)
    model_version = register_training_run(
        settings,
        paths,
        holdout_days=chosen_holdout,
        calibration_method=settings.phase3_calibration_method,
        blend_weight=float(artifact.logistic_weight),
        feature_frame=feat,
    )
    console.print(f"Train tamam -> {paths.model_path} (version={model_version})")


@app.command("predict")
def predict_cmd(
    target_date: str = typer.Option(None, "--date", help="YYYY-MM-DD"),
) -> None:
    settings = get_settings()
    paths = paths_from_settings(settings)
    if target_date:
        dt = date.fromisoformat(target_date)
    else:
        feat = build_features(paths).frame
        feat["date"] = feat["date"].astype(str)
        dt = date.fromisoformat(max(feat["date"]))

    pred = predict_for_date(
        paths,
        dt,
        enable_ev=settings.phase3_enable_ev,
        ev_probability_threshold=settings.ev_probability_threshold,
        ev_min_edge=settings.ev_min_edge,
        ev_min_value=settings.ev_min_value,
    )
    table = Table(title=f"Phase 3 Tahmin - {dt.isoformat()}")
    table.add_column("Race")
    table.add_column("Horse")
    table.add_column("P(win)")
    table.add_column("P(2.)")
    table.add_column("P(3.)")
    table.add_column("Edge")
    table.add_column("EV")
    table.add_column("Kelly")
    table.add_column("Decision")
    table.add_column("Rank")

    for _, row in pred.sort_values(["race_id", "rank"]).iterrows():
        table.add_row(
            str(row["race_id"]),
            str(row["horse_id"]),
            f"{row['calibrated_probability']:.3f}",
            f"{row.get('place2_probability', 0.0):.3f}",
            f"{row.get('place3_probability', 0.0):.3f}",
            f"{row.get('edge', 0.0):.3f}",
            f"{row.get('ev', 0.0):.3f}",
            f"{row.get('kelly_fraction', 0.0):.3f}",
            str(row.get("bet_decision", "-")),
            str(int(row["rank"])),
        )

    console.print(table)


@app.command("backtest")
def backtest_cmd(
    min_train_days: int = typer.Option(None, "--min-train-days"),
) -> None:
    settings = get_settings()
    paths = paths_from_settings(settings)
    chosen_min_train = min_train_days if min_train_days is not None else settings.phase1_min_train_days
    result = run_phase1_backtest(
        paths,
        min_train_days=chosen_min_train,
        calibration_days=settings.phase3_calibration_days,
        calibration_method=settings.phase3_calibration_method,
        ev_probability_threshold=settings.ev_probability_threshold,
        ev_min_edge=settings.ev_min_edge,
        ev_min_value=settings.ev_min_value,
    )
    console.print_json(json.dumps(result, ensure_ascii=True))


@app.command("calibrate")
def calibrate_cmd() -> None:
    settings = get_settings()
    paths = paths_from_settings(settings)
    _ = train_phase1_model(
        paths,
        holdout_days=settings.phase1_holdout_days,
        calibration_days=settings.phase3_calibration_days,
        calibration_method=settings.phase3_calibration_method,
    )
    console.print(
        f"Kalibrasyon tamam. Method={settings.phase3_calibration_method}, gun={settings.phase3_calibration_days}"
    )


@app.command("optimize")
def optimize_cmd() -> None:
    optimize_ticket_cmd()


@app.command("optimize-ticket")
def optimize_ticket_cmd(
    budget: float = typer.Option(None, "--budget", help="Toplam kupon butcesi"),
    target_date: str = typer.Option(None, "--date", help="YYYY-MM-DD"),
) -> None:
    settings = get_settings()
    paths = paths_from_settings(settings)

    if target_date:
        dt = date.fromisoformat(target_date)
    else:
        feat = build_features(paths).frame
        feat["date"] = feat["date"].astype(str)
        dt = date.fromisoformat(max(feat["date"]))

    result = optimize_for_date(paths, dt, settings, budget=budget)
    summary = result["summary"]
    console.print(
        f"Optimize summary: status={summary['status']} budget={summary['budget']:.2f} spent={summary['spent']:.2f} columns={summary['column_count']}"
    )

    columns = result.get("columns", [])
    if not columns:
        if summary.get("reason"):
            console.print(f"Reason: {summary['reason']}")
        return

    table = Table(title=f"Phase 4 6'li Kolon Portfoyu - {dt.isoformat()}")
    table.add_column("ID")
    table.add_column("Strategy")
    table.add_column("Combination")
    table.add_column("Prob")
    table.add_column("EV")
    table.add_column("MC Hit")
    table.add_column("Cost")

    for c in columns:
        comb = ", ".join(f"{k}:{v}" for k, v in c["combination"].items())
        table.add_row(
            str(c["column_id"]),
            str(c["strategy"]),
            comb,
            f"{c['probability']:.6f}",
            f"{c['ev']:.3f}",
            f"{c['monte_carlo_hit_rate']:.6f}",
            f"{c['cost']:.2f}",
        )

    console.print(table)


@app.command("report")
def report_cmd(
    target_date: str = typer.Option(None, "--date", help="YYYY-MM-DD"),
    budget: float = typer.Option(None, "--budget", help="Rapor icin optimize butcesi"),
) -> None:
    settings = get_settings()
    paths = paths_from_settings(settings)

    if target_date:
        dt = date.fromisoformat(target_date)
    else:
        feat = build_features(paths).frame
        feat["date"] = feat["date"].astype(str)
        dt = date.fromisoformat(max(feat["date"]))

    result = run_phase5_report(settings, paths, target_date=dt, budget=budget)
    console.print(
        f"Report tamam: model={result['model_version']} html={result['report_paths']['html']} json={result['report_paths']['json']}"
    )
