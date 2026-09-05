from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import atyaris.ml.modeling as modeling


def _minimal_payload() -> dict[str, object]:
    return {
        "stage1_model": object(),
        "stage2_model": object(),
        "feature_columns": ["f1", "f2"],
        "logistic_weight": 0.5,
        "calibrator": {"method": "none"},
    }


def test_safe_joblib_load_retries_on_numpy_core_module_error(monkeypatch, tmp_path: Path) -> None:
    calls = {"n": 0}

    def fake_load(path: str):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 1:
            raise ModuleNotFoundError("No module named 'numpy._core'", name="numpy._core")
        return _minimal_payload()

    monkeypatch.setattr(modeling.joblib, "load", fake_load)

    artifact, calibrator = modeling.load_phase3_artifact(str(tmp_path / "model.joblib"))

    assert calls["n"] == 2
    assert artifact.feature_columns == ["f1", "f2"]
    assert np.isclose(artifact.logistic_weight, 0.5)
    assert calibrator == {"method": "none"}


def test_safe_joblib_load_reraises_other_module_errors(monkeypatch, tmp_path: Path) -> None:
    def fake_load(path: str):  # type: ignore[no-untyped-def]
        raise ModuleNotFoundError("No module named 'x.y'", name="x.y")

    monkeypatch.setattr(modeling.joblib, "load", fake_load)

    try:
        modeling.load_phase3_artifact(str(tmp_path / "model.joblib"))
    except ModuleNotFoundError as exc:
        assert exc.name == "x.y"
    else:
        raise AssertionError("Expected ModuleNotFoundError to be raised")
