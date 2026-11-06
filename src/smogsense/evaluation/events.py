"""smogsense.evaluation.events — Exceedance and category verification.

Brier score of P(y > threshold) derived from the quantile function; POD, FAR, CSI, Peirce skill
score for 55.5, 125.5, 150.0 and 225.5 ug/m3, and ordinal AQI category accuracy.

Specification: docs/evaluation-strategy.md → 'Complementary metrics (calibration, events)'
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from smogsense.models.distribution import QuantileFunction
from smogsense.publishing.bulletin import determine_aqi_category

# Canonical PM2.5 breakpoints in ug/m3
THRESHOLD_UNHEALTHY = 55.5
THRESHOLD_VERY_UNHEALTHY = 125.5
THRESHOLD_SEVERE_SMOG = 150.0
THRESHOLD_HAZARDOUS = 225.5
THRESHOLD_HAZARDOUS_EMERGENCY = 250.0
CANONICAL_THRESHOLDS: tuple[float, ...] = (
    THRESHOLD_UNHEALTHY,
    THRESHOLD_VERY_UNHEALTHY,
    THRESHOLD_SEVERE_SMOG,
    THRESHOLD_HAZARDOUS,
    THRESHOLD_HAZARDOUS_EMERGENCY,
)

# Ordinal ranking of AQI categories for within-one-category accuracy
AQI_CATEGORY_RANKS: dict[str, int] = {
    "good": 0,
    "moderate": 1,
    "usg": 2,
    "unhealthy": 3,
    "very_unhealthy": 4,
    "hazardous": 5,
}


@dataclass(frozen=True)
class ContingencyTable:
    """Contingency matrix and classification verification diagnostics."""

    tp: int
    fp: int
    fn: int
    tn: int
    pod: float
    pofd: float
    far: float
    precision: float
    specificity: float
    csi: float
    pss: float
    f1: float
    accuracy: float
    decision_threshold: float
    exceedance_threshold: float

    def to_dict(self) -> dict[str, Any]:
        """Convert contingency table metrics to dictionary format."""
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "pod": self.pod,
            "pofd": self.pofd,
            "far": self.far,
            "precision": self.precision,
            "specificity": self.specificity,
            "csi": self.csi,
            "pss": self.pss,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "decision_threshold": self.decision_threshold,
            "exceedance_threshold": self.exceedance_threshold,
        }


def compute_exceedance_probabilities(
    quantiles_19: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Compute exceedance probabilities P(y > threshold) from 19-quantile predictions.

    Args:
        quantiles_19: (19,) or (N, 19) array of predicted quantiles.
        threshold: concentration threshold in ug/m3.

    Returns:
        Array of probabilities in [0, 1].
    """
    q_mat = np.atleast_2d(quantiles_19)
    if q_mat.shape[1] != 19:
        raise ValueError("quantiles_19 must have exactly 19 columns.")

    n_samples = q_mat.shape[0]
    probs = np.zeros(n_samples, dtype=float)

    for i in range(n_samples):
        qf = QuantileFunction(q_mat[i])
        val = qf.prob_exceed(float(threshold))
        probs[i] = float(np.asarray(val).item())

    probs = np.clip(probs, 0.0, 1.0)
    if quantiles_19.ndim == 1:
        return probs[0:1]
    return probs


def brier_score(
    y_true: Sequence[float] | np.ndarray,
    p_pred: Sequence[float] | np.ndarray,
    threshold: float | None = None,
) -> float:
    """Compute Brier Score for probabilistic binary event forecasts.

    Formula: BS = (1 / N) * sum((p_i - o_i)^2)

    Args:
        y_true: Array of true observations or binary outcomes {0, 1}.
        p_pred: Array of predicted event probabilities in [0, 1].
        threshold: If provided, observed event indicator is o_i = 1{y_i > threshold}.
                   If None, y_true is treated directly as binary indicators.

    Returns:
        Scalar Brier Score in [0, 1].
    """
    y_arr = np.asarray(y_true, dtype=float)
    p_arr = np.asarray(p_pred, dtype=float)

    if len(y_arr) != len(p_arr):
        raise ValueError(
            f"Observation length ({len(y_arr)}) must match probability length ({len(p_arr)})."
        )
    if len(y_arr) == 0:
        return 0.0

    if threshold is not None:
        o_arr = (y_arr > float(threshold)).astype(float)
    else:
        o_arr = (y_arr > 0.0).astype(float)

    return float(np.mean((p_arr - o_arr) ** 2))


def brier_skill_score(
    bs_model: float,
    bs_ref: float,
) -> float:
    """Compute Brier Skill Score (BSS) relative to reference baseline (e.g. climatology M3).

    Formula: BSS = 1.0 - (BS_model / BS_ref).

    Args:
        bs_model: Brier score of evaluated model.
        bs_ref: Brier score of reference baseline.

    Returns:
        BSS scalar. Returns 1.0 if both are 0.0; -inf if ref is 0.0 and model is > 0.
    """
    if bs_ref <= 0.0:
        return 1.0 if bs_model <= 0.0 else float("-inf")
    return float(1.0 - (bs_model / bs_ref))


def compute_contingency_table(
    y_true: Sequence[float] | np.ndarray,
    p_pred: Sequence[float] | np.ndarray,
    threshold: float = THRESHOLD_VERY_UNHEALTHY,
    decision_threshold: float = 0.50,
) -> ContingencyTable:
    """Compute 2x2 contingency table and standard skill scores for deterministic warning rule.

    Deterministic rule: "Warn if p >= decision_threshold".

    Metrics computed:
        - POD: Probability of Detection (Sensitivity / Recall) = TP / (TP + FN)
        - POFD: Probability of False Detection = FP / (FP + TN)
        - FAR: False Alarm Ratio = FP / (TP + FP)
        - Precision: TP / (TP + FP) = 1 - FAR
        - Specificity: TN / (TN + FP) = 1 - POFD
        - CSI: Critical Success Index (Threat Score) = TP / (TP + FP + FN)
        - PSS: Peirce Skill Score (Hanssen-Kuipers) = POD - POFD
        - F1: 2*TP / (2*TP + FP + FN)
        - Accuracy: (TP + TN) / (TP + FP + FN + TN)

    Args:
        y_true: Observed PM2.5 concentrations in ug/m3.
        p_pred: Predicted exceedance probabilities P(y > threshold).
        threshold: Smog exceedance threshold in ug/m3.
        decision_threshold: Probability threshold for warning issuance (default 0.50).

    Returns:
        ContingencyTable dataclass instance.
    """
    y_arr = np.asarray(y_true, dtype=float)
    p_arr = np.asarray(p_pred, dtype=float)

    if len(y_arr) != len(p_arr):
        raise ValueError(
            f"Observation length ({len(y_arr)}) must match probability length ({len(p_arr)})."
        )

    obs_events = y_arr > float(threshold)
    pred_events = p_arr >= float(decision_threshold)

    tp = int(np.sum(pred_events & obs_events))
    fp = int(np.sum(pred_events & (~obs_events)))
    fn = int(np.sum((~pred_events) & obs_events))
    tn = int(np.sum((~pred_events) & (~obs_events)))

    total = tp + fp + fn + tn

    pod = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    pofd = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    far = float(fp / (tp + fp)) if (tp + fp) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    csi = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0
    pss = float(pod - pofd)
    f1 = float(2.0 * tp / (2.0 * tp + fp + fn)) if (2.0 * tp + fp + fn) > 0 else 0.0
    accuracy = float((tp + tn) / total) if total > 0 else 0.0

    return ContingencyTable(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        pod=pod,
        pofd=pofd,
        far=far,
        precision=precision,
        specificity=specificity,
        csi=csi,
        pss=pss,
        f1=f1,
        accuracy=accuracy,
        decision_threshold=float(decision_threshold),
        exceedance_threshold=float(threshold),
    )


def category_accuracy(
    y_true: Sequence[float] | np.ndarray,
    q50: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Compute AQI category verification accuracy: exact match and within-one-category.

    Evaluates whether the median prediction q50 falls in the same or adjacent ordinal AQI category
    as the true observation y_true.

    Args:
        y_true: Array of observed concentrations.
        q50: Array of predicted median concentrations.

    Returns:
        Dict with "exact_match" and "within_one_category" fractions.
    """
    y_arr = np.asarray(y_true, dtype=float)
    m_arr = np.asarray(q50, dtype=float)

    if len(y_arr) != len(m_arr):
        raise ValueError(
            f"Observation length ({len(y_arr)}) must match median length ({len(m_arr)})."
        )
    if len(y_arr) == 0:
        return {"exact_match": 0.0, "within_one_category": 0.0}

    exact_matches = 0
    within_one = 0
    n = len(y_arr)

    for i in range(n):
        obs_cat = determine_aqi_category(float(y_arr[i]))
        pred_cat = determine_aqi_category(float(m_arr[i]))

        obs_rank = AQI_CATEGORY_RANKS.get(obs_cat, 0)
        pred_rank = AQI_CATEGORY_RANKS.get(pred_cat, 0)

        diff = abs(obs_rank - pred_rank)
        if diff == 0:
            exact_matches += 1
        if diff <= 1:
            within_one += 1

    return {
        "exact_match": float(exact_matches / n),
        "within_one_category": float(within_one / n),
    }


__all__ = [
    "AQI_CATEGORY_RANKS",
    "CANONICAL_THRESHOLDS",
    "THRESHOLD_HAZARDOUS",
    "THRESHOLD_HAZARDOUS_EMERGENCY",
    "THRESHOLD_SEVERE_SMOG",
    "THRESHOLD_UNHEALTHY",
    "THRESHOLD_VERY_UNHEALTHY",
    "ContingencyTable",
    "brier_score",
    "brier_skill_score",
    "category_accuracy",
    "compute_contingency_table",
    "compute_exceedance_probabilities",
]
