import unittest

import pandas as pd

from churn_analysis import (
    REQUIRED_COLUMNS,
    clean_customer_data,
    evaluate_binary_classifier,
)


def valid_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {column: "No" for column in REQUIRED_COLUMNS}
    row.update(
        {
            "customerID": "test-customer",
            "SeniorCitizen": 0,
            "tenure": 1,
            "MonthlyCharges": 10.0,
            "TotalCharges": "10.0",
            "Churn": "No",
        }
    )
    row.update(updates)
    return row


class ChurnAnalysisTests(unittest.TestCase):
    def test_removes_blank_total_charges_and_converts_numeric_values(self) -> None:
        raw = pd.DataFrame(
            [valid_row(TotalCharges=" "), valid_row(TotalCharges="25.50")]
        )

        cleaned = clean_customer_data(raw)

        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned["TotalCharges"].iloc[0], 25.5)
        self.assertEqual(raw.loc[0, "TotalCharges"], " ")

    def test_rejects_unexpected_churn_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "Churn labels"):
            clean_customer_data(pd.DataFrame([valid_row(Churn="Unknown")]))

    def test_rejects_non_numeric_total_charges(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-numeric"):
            clean_customer_data(pd.DataFrame([valid_row(TotalCharges="unknown")]))

    def test_rejects_missing_required_columns(self) -> None:
        data = pd.DataFrame([{"Churn": "No", "TotalCharges": "10"}])

        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            clean_customer_data(data)

    def test_rejects_missing_values_outside_total_charges(self) -> None:
        with self.assertRaisesRegex(ValueError, "other than TotalCharges"):
            clean_customer_data(pd.DataFrame([valid_row(gender=None)]))

    def test_lower_threshold_trades_precision_for_recall(self) -> None:
        labels = pd.Series([0, 1, 1, 0])
        scores = pd.Series([0.2, 0.9, 0.4, 0.3])

        default_metrics, _ = evaluate_binary_classifier(labels, scores.to_numpy())
        lower_metrics, lower_predictions = evaluate_binary_classifier(
            labels, scores.to_numpy(), threshold=0.3
        )

        self.assertGreater(lower_metrics["Recall"], default_metrics["Recall"])
        self.assertLess(lower_metrics["Precision"], default_metrics["Precision"])
        self.assertEqual(lower_predictions.tolist(), [0, 1, 1, 1])

    def test_rejects_invalid_probability_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            evaluate_binary_classifier(
                pd.Series([0, 1]), pd.Series([0.2, 0.8]).to_numpy(), threshold=1.1
            )


if __name__ == "__main__":
    unittest.main()
