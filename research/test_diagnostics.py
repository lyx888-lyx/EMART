#!/usr/bin/env python3
"""CPU-only tests for research/diagnose_predictions.py."""
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("diagnose_predictions.py")
spec = importlib.util.spec_from_file_location("diagnose_predictions", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DiagnosticsTests(unittest.TestCase):
    def test_metrics_and_false_neutral(self):
        labels = module.LABELS["meld7"]
        rows = {
            "a": {"instance": "a", "true": "neutral", "pred": "neutral"},
            "b": {"instance": "b", "true": "disgust", "pred": "neutral"},
            "c": {"instance": "c", "true": "disgust", "pred": "disgust"},
        }
        summary, per_class, matrix, errors = module.calculate(rows, labels)
        self.assertEqual(summary["n_valid"], 3)
        self.assertAlmostEqual(summary["accuracy"], 2 / 3)
        self.assertEqual(summary["false_neutral_count"], 1)
        self.assertAlmostEqual(summary["false_neutral_rate_given_non_neutral"], .5)
        self.assertEqual(matrix["disgust", "neutral"], 1)
        self.assertEqual(errors[0]["instance"], "b")

    def test_pairwise_alignment(self):
        labels = module.LABELS["meld7"]
        base = {"a": {"instance": "a", "true": "neutral", "pred": "neutral"},
                "b": {"instance": "b", "true": "disgust", "pred": "neutral"}}
        candidate = {"a": {"instance": "a", "true": "neutral", "pred": "disgust"},
                     "b": {"instance": "b", "true": "disgust", "pred": "disgust"}}
        with tempfile.TemporaryDirectory() as tmp:
            summary = module.paired_comparison(base, candidate, labels, Path(tmp))
            self.assertEqual(summary["baseline_wrong_candidate_right"], 1)
            self.assertEqual(summary["baseline_right_candidate_wrong"], 1)
            self.assertEqual(summary["paired_accuracy_delta"], 0)

    def test_reads_existing_csv_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preds.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["instance", "ground_truth_idx", "pred_idx"])
                w.writeheader()
                w.writerow(dict(instance="x", ground_truth_idx=6, pred_idx=0))
            rows, skipped = module.read_rows(path, "meld7")
            self.assertEqual(rows["x"]["true"], "disgust")
            self.assertEqual(rows["x"]["pred"], "neutral")
            self.assertFalse(skipped)


if __name__ == "__main__":
    unittest.main()
