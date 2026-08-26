from __future__ import annotations

from datetime import date, timedelta
import json

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from atyaris.config import get_settings
from atyaris.ml.phase5 import register_training_run, run_phase5_report
from atyaris.ml.pipeline import (
    build_features,
    ingest_synthetic,
    optimize_for_date,
    paths_from_settings,
    predict_for_date,
    preprocess_raw,
    run_phase1_backtest,
    train_phase1_model,
)

app = typer.Typer(add_completion=False, help="Phase 5 ML pipeline commands")
console = Console()


@app.command("ingest")
def ingest_cmd(
    start_date: str = typer.Option("2024-01-01", "--start-date"),
    end_date: str = typer.Option("2025-12-31", "--end-date"),
) -> None:
    settings = get_settings()
    paths = paths_from_settings(settings)
    data = ingest_synthetic(date.fromisoformat(start_date), date.fromisoformat(end_date), paths)
    console.print(f"Ingest tamam: {len(data)} satir -> {paths.raw_csv}")


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
    table.add_column("Edge")
    table.add_column("EV")
    table.add_column("Decision")
    table.add_column("Rank")

    for _, row in pred.sort_values(["race_id", "rank"]).iterrows():
        table.add_row(
            str(row["race_id"]),
            str(row["horse_id"]),
            f"{row['calibrated_probability']:.3f}",
            f"{row.get('edge', 0.0):.3f}",
            f"{row.get('ev', 0.0):.3f}",
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
