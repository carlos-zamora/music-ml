import numpy as np
import pytest
from evaluation import (
    compute_multilabel_metrics,
    compute_per_playlist_confusion,
    predict_labels,
    tune_thresholds,
)

EXPECTED_METRIC_KEYS = {
    "micro_f1", "macro_f1",
    "micro_precision", "macro_precision",
    "micro_recall", "macro_recall",
}

EXPECTED_ROW_KEYS = {
    "playlist", "support", "threshold",
    "precision", "recall", "f1",
    "tp", "fp", "tn", "fn",
}


# --- predict_labels ---

def test_predict_labels_basic():
    y_prob = np.array([[0.6]])
    result = predict_labels(y_prob, thresholds=0.5)
    assert result[0, 0] == 1.0


def test_predict_labels_per_class():
    y_prob = np.array([[0.6, 0.3]])
    # Class 0: threshold 0.5 → 1; Class 1: threshold 0.5 → 0
    result = predict_labels(y_prob, thresholds=[0.5, 0.5])
    assert result[0, 0] == 1.0
    assert result[0, 1] == 0.0


def test_predict_labels_length_mismatch_raises():
    y_prob = np.array([[0.6, 0.4]])  # 2 classes
    with pytest.raises(ValueError):
        predict_labels(y_prob, thresholds=[0.5, 0.5, 0.5])  # 3 thresholds


# --- compute_multilabel_metrics ---

def test_metrics_perfect_predictions():
    y_true = np.array([[1, 0], [0, 1]])
    y_pred = np.array([[1, 0], [0, 1]])
    metrics = compute_multilabel_metrics(y_true, y_pred)
    for key in EXPECTED_METRIC_KEYS:
        assert metrics[key] == pytest.approx(1.0), f"{key} should be 1.0"


def test_metrics_all_zero_pred_no_crash():
    y_true = np.array([[1, 0], [0, 1]])
    y_pred = np.zeros((2, 2))
    metrics = compute_multilabel_metrics(y_true, y_pred)
    for key in EXPECTED_METRIC_KEYS:
        assert metrics[key] == pytest.approx(0.0), f"{key} should be 0.0"


def test_metrics_returns_expected_keys():
    y_true = np.array([[1, 0]])
    y_pred = np.array([[1, 0]])
    metrics = compute_multilabel_metrics(y_true, y_pred)
    assert set(metrics.keys()) == EXPECTED_METRIC_KEYS


# --- compute_per_playlist_confusion ---

def test_confusion_counts():
    # Class A: true=[1,0,1], pred=[1,0,0] → TP=1, FP=0, TN=1, FN=1
    # Class B: true=[0,1,1], pred=[0,0,1] → TP=1, FP=0, TN=1, FN=1
    y_true = np.array([[1, 0], [0, 1], [1, 1]])
    y_pred = np.array([[1, 0], [0, 0], [0, 1]])
    rows = compute_per_playlist_confusion(y_true, y_pred, ["A", "B"])

    row_a = next(r for r in rows if r["playlist"] == "A")
    assert row_a["tp"] == 1
    assert row_a["fp"] == 0
    assert row_a["tn"] == 1
    assert row_a["fn"] == 1

    row_b = next(r for r in rows if r["playlist"] == "B")
    assert row_b["tp"] == 1
    assert row_b["fp"] == 0
    assert row_b["tn"] == 1
    assert row_b["fn"] == 1


def test_confusion_row_keys():
    y_true = np.array([[1, 0]])
    y_pred = np.array([[1, 0]])
    rows = compute_per_playlist_confusion(y_true, y_pred, ["A", "B"])
    for row in rows:
        assert EXPECTED_ROW_KEYS.issubset(set(row.keys()))


# --- tune_thresholds ---

def test_tune_thresholds_length():
    y_true = np.array([[1, 0], [0, 1]])
    y_prob = np.array([[0.8, 0.2], [0.3, 0.7]])
    thresholds = tune_thresholds(y_true, y_prob)
    assert len(thresholds) == 2


def test_tune_thresholds_tie_breaks_to_half():
    # All thresholds ≤ 0.5 give perfect F1; 0.5 is closest to default.
    y_true = np.array([[1], [1]])
    y_prob = np.array([[0.5], [0.5]])
    thresholds = tune_thresholds(y_true, y_prob)
    assert thresholds[0] == pytest.approx(0.5, abs=0.01)


def test_tune_thresholds_all_negative_class():
    # No positive examples in class → default threshold returned.
    y_true = np.array([[0], [0]])
    y_prob = np.array([[0.8], [0.3]])
    thresholds = tune_thresholds(y_true, y_prob, default_threshold=0.5)
    assert thresholds[0] == pytest.approx(0.5)
