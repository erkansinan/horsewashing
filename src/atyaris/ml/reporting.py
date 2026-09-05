from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _sparkline_svg(values: list[float], width: int = 520, height: int = 120, color: str = "#0b3a6e") -> str:
  if not values:
    return "<p>No trend data.</p>"
  if len(values) == 1:
    v = values[0]
    return (
      f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg'>"
      f"<line x1='0' y1='{height/2:.1f}' x2='{width}' y2='{height/2:.1f}' stroke='#dbe2ea' stroke-width='1'/>"
      f"<circle cx='{width/2:.1f}' cy='{height/2:.1f}' r='4' fill='{color}'/>"
      f"<text x='8' y='18' fill='{color}' font-size='12'>{v:.4f}</text></svg>"
    )

  min_v = min(values)
  max_v = max(values)
  span = max(max_v - min_v, 1e-12)

  points = []
  for i, v in enumerate(values):
    x = (i / (len(values) - 1)) * (width - 16) + 8
    y = height - 8 - ((v - min_v) / span) * (height - 20)
    points.append((x, y))

  polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
  return (
    f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg'>"
    f"<rect x='0' y='0' width='{width}' height='{height}' fill='white'/>"
    f"<line x1='8' y1='{height-8}' x2='{width-8}' y2='{height-8}' stroke='#dbe2ea' stroke-width='1'/>"
    f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{polyline}'/>"
    f"<text x='8' y='14' fill='#334155' font-size='11'>min={min_v:.4f}</text>"
    f"<text x='{width-120}' y='14' fill='#334155' font-size='11'>max={max_v:.4f}</text>"
    "</svg>"
  )


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def generate_phase5_report(
    report_dir: str,
    *,
    model_version: str,
    target_date: str,
    backtest_metrics: dict[str, Any],
    prediction_frame: pd.DataFrame,
    optimization_result: dict[str, Any],
    explainability: dict[str, Any],
    tracking_snapshot: dict[str, Any],
) -> dict[str, str]:
    out_dir = Path(report_dir)
    _ensure_dir(out_dir)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"phase5_report_{timestamp}.json"
    html_path = out_dir / f"phase5_dashboard_{timestamp}.html"

    top_preds = (
        prediction_frame.sort_values(["race_id", "rank"]).groupby("race_id").head(3).copy()
        if not prediction_frame.empty
        else pd.DataFrame()
    )

    payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "model_version": model_version,
        "target_date": target_date,
        "backtest_metrics": backtest_metrics,
        "optimization_summary": optimization_result.get("summary", {}),
        "columns": optimization_result.get("columns", []),
        "top_predictions": top_preds.to_dict(orient="records") if not top_preds.empty else [],
        "explainability": explainability,
        "tracking_snapshot": tracking_snapshot,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8")

    table_cols = [
      "race_id",
      "horse_id",
      "calibrated_probability",
      "place2_probability",
      "place3_probability",
      "top3_probability",
      "edge",
      "ev",
      "kelly_fraction",
      "bet_decision",
      "confidence",
    ]
    existing_cols = [c for c in table_cols if c in top_preds.columns]
    top_table_html = top_preds[existing_cols].to_html(index=False) if not top_preds.empty else "<p>Prediction data unavailable.</p>"

    cols = optimization_result.get("columns", [])
    cols_df = pd.DataFrame(cols)
    if not cols_df.empty and "combination" in cols_df.columns:
        cols_df = cols_df.copy()
        cols_df["combination"] = cols_df["combination"].apply(lambda d: ", ".join(f"{k}:{v}" for k, v in d.items()))
    cols_html = cols_df.to_html(index=False) if not cols_df.empty else "<p>No columns generated (NO_BET).</p>"

    feature_rows = explainability.get("permutation_top", [])
    feat_df = pd.DataFrame(feature_rows)
    feat_html = feat_df.to_html(index=False) if not feat_df.empty else "<p>No explainability data.</p>"

    fold_log_loss = [float(x) for x in backtest_metrics.get("fold_log_loss", [])]
    fold_brier = [float(x) for x in backtest_metrics.get("fold_brier", [])]
    fold_roi = [float(x) for x in backtest_metrics.get("fold_roi", [])]
    chart_log_loss = _sparkline_svg(fold_log_loss, color="#0b3a6e")
    chart_brier = _sparkline_svg(fold_brier, color="#0f766e")
    chart_roi = _sparkline_svg(fold_roi, color="#9a3412")

    html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Phase 5 Dashboard</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; background: #f8fafc; color: #0f172a; }}
    h1, h2 {{ color: #0b3a6e; }}
    .card {{ background: white; border: 1px solid #dbe2ea; border-radius: 10px; padding: 14px 16px; margin-bottom: 14px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #dbe2ea; padding: 6px 8px; text-align: left; }}
    th {{ background: #eef4fb; }}
    .meta {{ display: grid; grid-template-columns: repeat(3, minmax(180px, 1fr)); gap: 10px; }}
  </style>
</head>
<body>
  <h1>Phase 5 Dashboard</h1>
  <div class=\"card meta\">
    <div><strong>Model Version</strong><br>{model_version}</div>
    <div><strong>Target Date</strong><br>{target_date}</div>
    <div><strong>Generated</strong><br>{payload['generated_at']}</div>
  </div>

  <h2>Backtest Summary</h2>
  <div class=\"card\"><pre>{json.dumps(backtest_metrics, ensure_ascii=True, indent=2, default=str)}</pre></div>

  <h2>Fold Trends</h2>
  <div class=\"card\"><strong>Log Loss (by fold)</strong><br>{chart_log_loss}</div>
  <div class=\"card\"><strong>Brier Score (by fold)</strong><br>{chart_brier}</div>
  <div class=\"card\"><strong>ROI (by fold)</strong><br>{chart_roi}</div>

  <h2>Top Predictions (Top-3 per Race)</h2>
  <div class=\"card\">{top_table_html}</div>

  <h2>Ticket Portfolio</h2>
  <div class=\"card\"><pre>{json.dumps(optimization_result.get('summary', {}), ensure_ascii=True, indent=2, default=str)}</pre>{cols_html}</div>

  <h2>Explainability (Permutation Importance)</h2>
  <div class=\"card\">{feat_html}</div>

  <h2>Tracking Snapshot</h2>
  <div class=\"card\"><pre>{json.dumps(tracking_snapshot, ensure_ascii=True, indent=2, default=str)}</pre></div>
</body>
</html>
""".strip()

    html_path.write_text(html, encoding="utf-8")
    return {"json": str(json_path), "html": str(html_path)}
