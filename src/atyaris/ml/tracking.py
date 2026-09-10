from __future__ import annotations

from datetime import datetime
import json
import sqlite3
from pathlib import Path
from typing import Any


def init_tracking_db(db_path: str) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                artifact_path TEXT NOT NULL,
                train_start_date TEXT,
                train_end_date TEXT,
                holdout_days INTEGER,
                calibration_method TEXT,
                blend_weight REAL,
                metrics_json TEXT,
                status TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS backtest_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metrics_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ticket_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                target_date TEXT NOT NULL,
                budget REAL NOT NULL,
                summary_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT NOT NULL UNIQUE,
                is_production INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def generate_model_version(prefix: str = "model_v") -> str:
    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}{now}"


def register_model_run(
    db_path: str,
    *,
    model_version: str,
    artifact_path: str,
    train_start_date: str | None,
    train_end_date: str | None,
    holdout_days: int,
    calibration_method: str,
    blend_weight: float,
    metrics: dict[str, Any],
    status: str = "candidate",
) -> int:
    init_tracking_db(db_path)
    now = datetime.utcnow().isoformat()
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO model_runs (
                model_version, created_at, artifact_path,
                train_start_date, train_end_date, holdout_days,
                calibration_method, blend_weight, metrics_json, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_version,
                now,
                artifact_path,
                train_start_date,
                train_end_date,
                holdout_days,
                calibration_method,
                blend_weight,
                json.dumps(metrics, ensure_ascii=True),
                status,
            ),
        )
        conn.execute(
            "INSERT OR IGNORE INTO model_registry (model_version, is_production, updated_at) VALUES (?, 0, ?)",
            (model_version, now),
        )
        conn.commit()
        return int(cur.lastrowid)


def mark_model_production(db_path: str, model_version: str) -> None:
    init_tracking_db(db_path)
    now = datetime.utcnow().isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE model_registry SET is_production = 0, updated_at = ?", (now,))
        conn.execute(
            "INSERT OR IGNORE INTO model_registry (model_version, is_production, updated_at) VALUES (?, 1, ?)",
            (model_version, now),
        )
        conn.execute(
            "UPDATE model_registry SET is_production = 1, updated_at = ? WHERE model_version = ?",
            (now, model_version),
        )
        conn.commit()


def log_backtest_run(db_path: str, model_version: str, metrics: dict[str, Any]) -> None:
    init_tracking_db(db_path)
    now = datetime.utcnow().isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO backtest_runs (model_version, created_at, metrics_json) VALUES (?, ?, ?)",
            (model_version, now, json.dumps(metrics, ensure_ascii=True)),
        )
        conn.commit()


def log_ticket_run(db_path: str, model_version: str, target_date: str, budget: float, summary: dict[str, Any]) -> None:
    init_tracking_db(db_path)
    now = datetime.utcnow().isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO ticket_runs (model_version, created_at, target_date, budget, summary_json) VALUES (?, ?, ?, ?, ?)",
            (model_version, now, target_date, budget, json.dumps(summary, ensure_ascii=True)),
        )
        conn.commit()


def latest_model_version(db_path: str) -> str | None:
    init_tracking_db(db_path)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT model_version FROM model_runs ORDER BY id DESC LIMIT 1").fetchone()
        return None if row is None else str(row[0])


def load_recent_runs(db_path: str, limit: int = 10) -> dict[str, Any]:
    init_tracking_db(db_path)
    out: dict[str, Any] = {"models": [], "backtests": [], "tickets": []}
    with sqlite3.connect(db_path) as conn:
        model_rows = conn.execute(
            "SELECT model_version, created_at, status, calibration_method, blend_weight FROM model_runs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        out["models"] = [
            {
                "model_version": r[0],
                "created_at": r[1],
                "status": r[2],
                "calibration_method": r[3],
                "blend_weight": r[4],
            }
            for r in model_rows
        ]

        bt_rows = conn.execute(
            "SELECT model_version, created_at, metrics_json FROM backtest_runs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        out["backtests"] = [
            {
                "model_version": r[0],
                "created_at": r[1],
                "metrics": json.loads(r[2]),
            }
            for r in bt_rows
        ]

        tk_rows = conn.execute(
            "SELECT model_version, created_at, target_date, budget, summary_json FROM ticket_runs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        out["tickets"] = [
            {
                "model_version": r[0],
                "created_at": r[1],
                "target_date": r[2],
                "budget": r[3],
                "summary": json.loads(r[4]),
            }
            for r in tk_rows
        ]
    return out


def load_training_history(
    db_path: str,
    *,
    limit: int = 10,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Return training history and changes relative to the previous run."""
    init_tracking_db(db_path)
    safe_limit = max(1, min(int(limit), 100))
    conditions = []
    params: list[Any] = []
    if start_date:
        conditions.append("train_end_date >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("train_start_date <= ?")
        params.append(end_date)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT id, model_version, created_at, artifact_path,
                   train_start_date, train_end_date, holdout_days,
                   calibration_method, blend_weight, metrics_json, status
            FROM model_runs
            {where}
            ORDER BY id DESC
            LIMIT ?
            """,
            [*params, safe_limit],
        ).fetchall()
        all_rows = conn.execute(
            """
            SELECT id, model_version, created_at, artifact_path,
                   train_start_date, train_end_date, holdout_days,
                   calibration_method, blend_weight, metrics_json, status
            FROM model_runs
            ORDER BY id DESC
            """
        ).fetchall()

    def _load_health_report(artifact_path: str) -> dict[str, Any]:
        health_path = Path(artifact_path).with_suffix(".health.json")
        try:
            payload = json.loads(health_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _row_to_run(row: tuple[Any, ...]) -> dict[str, Any]:
        try:
            metrics = json.loads(row[9] or "{}")
        except json.JSONDecodeError:
            metrics = {}
        return {
            "id": row[0],
            "model_version": row[1],
            "trained_at": row[2],
            "artifact_path": row[3],
            "train_start_date": row[4],
            "train_end_date": row[5],
            "training_interval": {"start": row[4], "end": row[5]},
            "holdout_days": row[6],
            "calibration_method": row[7],
            "blend_weight": row[8],
            "status": row[10],
            "metrics": metrics,
            "model_health": _load_health_report(row[3]),
        }

    runs: list[dict[str, Any]] = []
    newer_first = [_row_to_run(row) for row in rows]
    all_runs = [_row_to_run(row) for row in all_rows]
    previous_by_id = {
        run["id"]: all_runs[index + 1]
        for index, run in enumerate(all_runs[:-1])
    }

    for run in newer_first:
        previous = previous_by_id.get(run["id"])
        current_features = set(run["metrics"].get("feature_columns", []))
        previous_features = set(previous["metrics"].get("feature_columns", [])) if previous else set()
        run["changes_from_previous"] = {
            "previous_model_version": previous["model_version"] if previous else None,
            "added_features": sorted(current_features - previous_features) if previous else [],
            "removed_features": sorted(previous_features - current_features) if previous else [],
            "feature_count_change": (
                len(current_features) - len(previous_features) if previous else None
            ),
            "blend_weight_change": (
                run["blend_weight"] - previous["blend_weight"]
                if previous and run["blend_weight"] is not None and previous["blend_weight"] is not None
                else None
            ),
            "calibration_changed": (
                run["calibration_method"] != previous["calibration_method"] if previous else None
            ),
        }
        runs.append(run)

    return {
        "query": {"limit": safe_limit, "start_date": start_date, "end_date": end_date},
        "latest_training": runs[0] if runs else None,
        "runs": runs,
        "count": len(runs),
    }
