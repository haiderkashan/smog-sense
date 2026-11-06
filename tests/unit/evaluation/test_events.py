"""Unit tests for AQI category accuracy and probabilistic event exceedance verification."""

import numpy as np
import pytest
from scipy.stats import norm

from smogsense.evaluation.events import (
    CANONICAL_THRESHOLDS,
    THRESHOLD_HAZARDOUS,
    THRESHOLD_SEVERE_SMOG,
    THRESHOLD_UNHEALTHY,
    THRESHOLD_VERY_UNHEALTHY,
    brier_score,
    brier_skill_score,
    category_accuracy,
    compute_contingency_table,
    compute_exceedance_probabilities,
)
from smogsense.evaluation.scoring import CANONICAL_TAUS


def test_compute_exceedance_probabilities() -> None:
    # 19 quantiles from N(100, 20^2)
    q19 = norm.ppf(CANONICAL_TAUS, loc=100.0, scale=20.0)

    # Threshold at median (100.0) -> exceedance probability ~ 0.50
    p_100 = compute_exceedance_probabilities(q19, threshold=100.0)
    assert len(p_100) == 1
    assert np.isclose(p_100[0], 0.50, atol=0.03)

    # Threshold at high tail (150.0) -> exceedance probability should be low
    p_150 = compute_exceedance_probabilities(q19, threshold=150.0)
    assert p_150[0] < 0.05

    # Shape mismatch raises ValueError
    with pytest.raises(ValueError, match="exactly 19 columns"):
        compute_exceedance_probabilities(np.ones(10), threshold=100.0)


def test_brier_score_and_skill_score() -> None:
    # Binary outcomes and perfect forecasts
    y_bin = np.array([1.0, 0.0, 1.0, 0.0])
    p_perfect = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(y_bin, p_perfect) == 0.0

    # Constant 0.5 forecast: BS = 0.25
    p_climo = np.array([0.5, 0.5, 0.5, 0.5])
    assert np.isclose(brier_score(y_bin, p_climo), 0.25)

    # Continuous y with threshold
    y_cont = np.array([40.0, 130.0, 60.0, 180.0])
    p_pred = np.array([0.1, 0.9, 0.2, 0.8])
    # threshold = 125.5 -> events are [0, 1, 0, 1]
    # diffs: [0.1 - 0 = 0.1, 0.9 - 1 = -0.1, 0.2 - 0 = 0.2, 0.8 - 1 = -0.2]
    # sq diffs: [0.01, 0.01, 0.04, 0.04] -> mean = 0.10 / 4 = 0.025
    bs = brier_score(y_cont, p_pred, threshold=125.5)
    assert np.isclose(bs, 0.025)

    # BSS
    bss = brier_skill_score(bs_model=0.05, bs_ref=0.20)
    assert np.isclose(bss, 0.75)


def test_compute_contingency_table_metrics() -> None:
    # Construct synthetic cases:
    # TP: y=160, p=0.8 (threshold 125.5, decision 0.50)
    # FP: y=100, p=0.7
    # FN: y=140, p=0.3
    # TN: y=80,  p=0.2
    y_true = np.array([160.0, 100.0, 140.0, 80.0])
    p_pred = np.array([0.80, 0.70, 0.30, 0.20])

    table = compute_contingency_table(y_true, p_pred, threshold=125.5, decision_threshold=0.50)
    assert table.tp == 1
    assert table.fp == 1
    assert table.fn == 1
    assert table.tn == 1

    # POD = TP / (TP + FN) = 1 / 2 = 0.50
    assert np.isclose(table.pod, 0.50)
    # POFD = FP / (FP + TN) = 1 / 2 = 0.50
    assert np.isclose(table.pofd, 0.50)
    # FAR = FP / (TP + FP) = 1 / 2 = 0.50
    assert np.isclose(table.far, 0.50)
    # CSI = TP / (TP + FP + FN) = 1 / 3 = 0.3333
    assert np.isclose(table.csi, 1.0 / 3.0)
    # PSS = POD - POFD = 0.50 - 0.50 = 0.0
    assert np.isclose(table.pss, 0.0)
    # Accuracy = (TP + TN) / Total = 2 / 4 = 0.50
    assert np.isclose(table.accuracy, 0.50)

    d = table.to_dict()
    assert d["tp"] == 1
    assert d["csi"] == table.csi


def test_category_accuracy_exact_and_within_one() -> None:
    # AQI Categories:
    # 25.0 -> moderate (rank 1)
    # 40.0 -> moderate (rank 1)
    # 60.0 -> usg (rank 2)
    # 160.0 -> very_unhealthy (rank 4)
    # 260.0 -> hazardous (rank 5)

    # True: [moderate, moderate, usg, very_unhealthy]
    y_true = np.array([25.0, 35.0, 60.0, 160.0])

    # Pred median:
    # 1. 25.0 (moderate) -> exact match (diff 0)
    # 2. 60.0 (usg) -> diff 1 (moderate vs usg)
    # 3. 160.0 (very_unhealthy) -> diff 2 (usg vs very_unhealthy)
    # 4. 170.0 (very_unhealthy) -> exact match (diff 0)
    q50_pred = np.array([25.0, 60.0, 160.0, 170.0])

    acc = category_accuracy(y_true, q50_pred)
    # Exact: samples 1 and 4 -> 2 / 4 = 0.50
    assert np.isclose(acc["exact_match"], 0.50)
    # Within one: samples 1, 2, and 4 -> 3 / 4 = 0.75
    assert np.isclose(acc["within_one_category"], 0.75)


def test_canonical_threshold_constants() -> None:
    assert THRESHOLD_UNHEALTHY == 55.5
    assert THRESHOLD_VERY_UNHEALTHY == 125.5
    assert THRESHOLD_SEVERE_SMOG == 150.0
    assert THRESHOLD_HAZARDOUS == 225.5
    assert len(CANONICAL_THRESHOLDS) == 5
