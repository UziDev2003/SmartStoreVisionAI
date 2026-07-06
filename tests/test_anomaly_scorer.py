"""Tests for the AnomalyScorer (sklearn-based)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
import numpy as np
from types import SimpleNamespace

from src.analytics.anomaly_scorer import AnomalyScorer, FEATURE_KEYS


def _features(avg=10.0, peak=20.0, var=0.5, dist=200.0, turns=2, net=180.0,
              dur=10.0, dwell=5.0) -> dict:
    return {
        "avg_speed": avg, "peak_speed": peak, "motion_var": var,
        "total_distance": dist, "heading_change_count": turns,
        "net_displacement": net, "duration_seconds": dur, "dwell_seconds": dwell,
    }


def test_score_returns_none_when_not_trained():
    cfg = SimpleNamespace(enabled=True, model_path="x", n_estimators=50,
                          contamination=0.05, confidence_weight=0.5,
                          min_samples_to_train=10)
    a = AnomalyScorer(cfg)
    assert a.is_trained is False
    assert a.score(_features()) is None


def test_combine_passthrough_when_no_model():
    cfg = SimpleNamespace(enabled=True, model_path="x", n_estimators=50,
                          contamination=0.05, confidence_weight=0.5,
                          min_samples_to_train=10)
    a = AnomalyScorer(cfg)
    assert a.combine(0.6, _features()) == 0.6


@pytest.mark.skipif(not __import__("os").environ.get("RUN_SKLEARN_TESTS"),
                    reason="Optional sklearn training test (set RUN_SKLEARN_TESTS=1)")
def test_train_and_score_synthetic():
    from sklearn.ensemble import IsolationForest
    cfg = SimpleNamespace(enabled=True, model_path="models/test_iforest.joblib",
                          n_estimators=50, contamination=0.05, confidence_weight=0.5,
                          min_samples_to_train=20)
    a = AnomalyScorer(cfg)
    # 100 normal-ish samples
    data = []
    for i in range(100):
        data.append(_features(avg=10 + np.random.randn(),
                              peak=20 + np.random.randn(),
                              var=0.5 + np.random.randn() * 0.1))
    ok = a.train(data)
    assert ok is True
    s_normal = a.score(_features())
    s_anom = a.score(_features(avg=200, peak=400, var=5, dist=5000, turns=80, dwell=600))
    assert s_normal is not None and s_anom is not None
    assert 0.0 <= s_normal <= 1.0 and 0.0 <= s_anom <= 1.0
