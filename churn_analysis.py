from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

REQUIRED_COLUMNS = (
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
)

VALID_CHURN_LABELS = {"Yes", "No"}


def clean_customer_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate the Telco churn schema and remove rows without TotalCharges."""
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    cleaned = data.loc[:, REQUIRED_COLUMNS].copy()

    invalid_labels = cleaned["Churn"].isna() | ~cleaned["Churn"].isin(
        VALID_CHURN_LABELS
    )
    if invalid_labels.any():
        labels = sorted(cleaned.loc[invalid_labels, "Churn"].astype(str).unique())
        raise ValueError(f"Unexpected or missing Churn labels: {labels}")

    total_charges = cleaned["TotalCharges"].astype("string")
    blank_total_charges = total_charges.str.strip().eq("").fillna(False)
    cleaned.loc[blank_total_charges, "TotalCharges"] = pd.NA

    converted_total_charges = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    invalid_total_charges = cleaned["TotalCharges"].notna() & (
        converted_total_charges.isna()
    )
    if invalid_total_charges.any():
        raise ValueError("TotalCharges contains non-numeric values.")
    cleaned["TotalCharges"] = converted_total_charges

    missing_other = cleaned.drop(columns="TotalCharges").isna().any()
    missing_columns_with_values = missing_other[missing_other].index.tolist()
    if missing_columns_with_values:
        raise ValueError(
            "Missing values found in required columns other than TotalCharges: "
            f"{missing_columns_with_values}"
        )

    cleaned = cleaned.dropna(subset=["TotalCharges"]).copy()
    if cleaned.empty:
        raise ValueError("No rows remain after removing missing TotalCharges values.")

    return cleaned


def evaluate_binary_classifier(
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> tuple[dict[str, float], np.ndarray]:
    """Return threshold-based metrics and predictions for positive-class scores."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    labels = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("y_true and y_score must be one-dimensional and equal length.")
    if len(labels) == 0:
        raise ValueError("y_true and y_score cannot be empty.")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("y_true must contain both binary labels 0 and 1.")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("y_score must contain finite probabilities between 0 and 1.")

    predictions = (scores >= threshold).astype(int)
    metrics = {
        "Threshold": float(threshold),
        "Accuracy": float(accuracy_score(labels, predictions)),
        "Precision": float(precision_score(labels, predictions, zero_division=0)),
        "Recall": float(recall_score(labels, predictions, zero_division=0)),
        "F1 Score": float(f1_score(labels, predictions, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(labels, scores)),
        "Average Precision": float(average_precision_score(labels, scores)),
    }
    return metrics, predictions
