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
