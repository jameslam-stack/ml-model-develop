"""Tests for MLDeveloperPipeline (agent-data-ml-model workflow)."""
import numpy as np
import pandas as pd
import pytest
from ml_model_toolkit import MLDeveloperPipeline


def _make_data(n=100):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "age": rng.integers(18, 70, n),
        "salary": rng.normal(60000, 15000, n),
        "target": rng.integers(0, 2, n),
    })


def test_full_pipeline_runs():
    df = _make_data()
    p = MLDeveloperPipeline(df, target_col="target")
    p.run_full()
    assert "accuracy" in p.metrics
    assert 0.0 <= p.metrics["accuracy"] <= 1.0
    assert p.model is not None
    assert p.predictions is not None


def test_all_five_stages_exist():
    p = MLDeveloperPipeline(_make_data(), target_col="target")
    for stage in ("analyze_data", "preprocess", "build_model",
                  "evaluate_model", "save_model", "save_report"):
        assert hasattr(p, stage), f"missing stage: {stage}"


def test_build_model_random_forest():
    df = _make_data()
    p = MLDeveloperPipeline(df, target_col="target")
    p.preprocess()
    p.build_model(algorithm="random_forest")
    p.evaluate_model()
    assert "accuracy" in p.metrics


def test_missing_value_imputation():
    df = _make_data()
    df.loc[0, "salary"] = np.nan
    p = MLDeveloperPipeline(df, target_col="target")
    p.preprocess()
    assert not p.X_train["salary"].isnull().any()


def test_pyspark_backend_optional():
    # Without pyspark installed, pandas backend is used by default
    from ml_model_toolkit.backends import get_backend
    b = get_backend(_make_data())
    assert b.is_pandas
