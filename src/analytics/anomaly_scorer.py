"""
Anomaly Scorer (scikit-learn)
=============================
Trains an IsolationForest on per-track movement features and provides
a per-track anomaly score that is combined with the rule-based
detector confidence. Designed to suppress low-confidence false positives.

Usage:
    scorer = AnomalyScorer(config)
    scorer.train(features_list)  # list of feature dicts from normal traffic
    score = scorer.score(features_dict)
    if score is not None:
        final_conf = 0.5 * rule_conf + 0.5 * (1 - score)
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np

try:
    from sklearn.ensemble import IsolationForest  # type: ignore
    from joblib import dump, load  # type: ignore
    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    SKLEARN_AVAILABLE = False


# Ordered feature vector used for training/inference
FEATURE_KEYS: List[str] = [
    "avg_speed",
    "peak_speed",
    "motion_var",
    "total_distance",
    "heading_change_count",
    "net_displacement",
    "duration_seconds",
    "dwell_seconds",
]


def _vectorize(features: Dict[str, float]) -> np.ndarray:
    """Convert a feature dict to a fixed-order numpy vector."""
    return np.array([float(features.get(k, 0.0)) for k in FEATURE_KEYS], dtype=np.float32)


class AnomalyScorer:
    """
    Wraps an IsolationForest model. Falls back to a no-op if sklearn is
    not installed; in that case all `score` calls return None and the
    detector pipeline continues without anomaly gating.
    """
    def __init__(self, config):
        self.config = config
        self.model = None
        self.is_trained = False
        self.feature_buffer: List[Dict[str, float]] = []
        if SKLEARN_AVAILABLE:
            try:
                self.model = IsolationForest(
                    n_estimators=int(getattr(config, "n_estimators", 200)),
                    contamination=float(getattr(config, "contamination", 0.05)),
                    random_state=42,
                )
            except Exception:
                self.model = None

    def add_training_sample(self, features: Dict[str, float]) -> None:
        """Buffer a training sample; useful for online accumulation."""
        if not SKLEARN_AVAILABLE:
            return
        self.feature_buffer.append({k: float(features.get(k, 0.0)) for k in FEATURE_KEYS})

    def train(self, features_list: Optional[List[Dict[str, float]]] = None) -> bool:
        """
        Train the IsolationForest. If `features_list` is None, use the
        internal buffer. Returns True on successful training.
        """
        if not SKLEARN_AVAILABLE or self.model is None:
            return False
        data = features_list if features_list is not None else self.feature_buffer
        if len(data) < int(getattr(self.config, "min_samples_to_train", 100)):
            return False
        X = np.stack([_vectorize(f) for f in data], axis=0)
        try:
            self.model.fit(X)
            self.is_trained = True
            # Persist model
            model_path = Path(getattr(self.config, "model_path", "models/anomaly_iforest.joblib"))
            model_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                dump({"model": self.model, "feature_keys": FEATURE_KEYS}, str(model_path))
            except Exception:
                pass
            return True
        except Exception:
            return False

    def load(self) -> bool:
        """Load a previously persisted model from disk."""
        if not SKLEARN_AVAILABLE:
            return False
        model_path = Path(getattr(self.config, "model_path", "models/anomaly_iforest.joblib"))
        if not model_path.exists():
            return False
        try:
            blob = load(str(model_path))
            self.model = blob["model"] if isinstance(blob, dict) else blob
            self.is_trained = True
            return True
        except Exception:
            return False

    def score(self, features: Dict[str, float]) -> Optional[float]:
        """
        Return an anomaly score in [0, 1] where 1.0 is most anomalous.
        Returns None if the model isn't trained or sklearn is unavailable.
        """
        if not SKLEARN_AVAILABLE or self.model is None or not self.is_trained:
            return None
        try:
            X = _vectorize(features).reshape(1, -1)
            raw = self.model.decision_function(X)[0]   # higher = more normal
            # Map to [0,1] anomaly score: 0 = normal, 1 = anomalous
            # raw is typically in [-0.5, 0.5]; flip and squash
            anom = 1.0 / (1.0 + np.exp(5.0 * raw))  # sigmoid
            return float(np.clip(anom, 0.0, 1.0))
        except Exception:
            return None

    def combine(self, rule_confidence: float, features: Dict[str, float]) -> float:
        """
        Combine rule confidence with anomaly score.
        final = (1 - w) * rule_conf + w * (1 - anomaly_score)
        i.e., an anomalous track does NOT boost confidence (suppresses
        false positives where the rule matched but motion is normal).
        """
        w = float(getattr(self.config, "confidence_weight", 0.5))
        s = self.score(features)
        if s is None:
            return float(rule_confidence)
        return float((1.0 - w) * rule_confidence + w * (1.0 - s))
