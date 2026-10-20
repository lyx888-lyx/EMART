#!/usr/bin/env python3
"""CPU-only unit tests for IEMOCAP wheel geometry audit."""
import csv
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(__file__).with_name("iemocap_wheel_validity.py")
spec = importlib.util.spec_from_file_location("iemocap_wheel_validity", SOURCE)
wheel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wheel)


def make_vad_row(instance, split, emotion, v, a, d=3):
    return {"instance": instance, "split": split, "emotion": emotion,
            "valence": float(v), "activation": float(a), "dominance": float(d)}


class WheelGeometryTests(unittest.TestCase):
    def test_angle_wrap_and_relative_order(self):
        self.assertAlmostEqual(wheel.angle_distance("hap", "sad"), math.pi)
        self.assertAlmostEqual(wheel.angle_distance("ang", "fru"), math.pi / 8)
        self.assertLess(wheel.angle_distance("hap", "exc"),
                        wheel.angle_distance("hap", "sad"))

    def test_midrank_ties_and_correlation(self):
        self.assertEqual(wheel.midranks([2, 1, 2]), [2.5, 1, 2.5])
        self.assertAlmostEqual(wheel.pearson([1, 2, 3], [2, 4, 6]), 1)

    def test_instance_specific_prior_inversion(self):
        anchor = make_vad_row("A", "dev", "hap", 4, 3)
        near_prior = make_vad_row("B", "dev", "exc", 1, 1)
        far_prior = make_vad_row("C", "dev", "sad", 4.1, 3)
        self.assertEqual(wheel.sample_prior_inversion(
            anchor, near_prior, far_prior, ("valence", "activation")), 1)
        self.assertIsNone(wheel.sample_prior_inversion(
            anchor, near_prior, near_prior, ("valence", "activation")))

    def test_real_split_string_labels_and_integer_labels(self):
        self.assertEqual(wheel.parse_emotion_label("neu", "item"), 0)
        self.assertEqual(wheel.parse_emotion_label("fru", "item"), 2)
        self.assertEqual(wheel.parse_emotion_label("ang", "item"), 3)
        self.assertEqual(wheel.parse_emotion_label("hap", "item"), 4)
        self.assertEqual(wheel.parse_emotion_label("exc", "item"), 5)
        self.assertEqual(wheel.parse_emotion_label("sad", "item"), 1)
        self.assertEqual(wheel.parse_emotion_label("happy", "item"), 4)
        self.assertEqual(wheel.parse_emotion_label(4, "item"), 4)
        self.assertEqual(wheel.parse_emotion_label("4", "item"), 4)
        with self.assertRaises(ValueError):
            wheel.parse_emotion_label("xxx", "item")
        with self.assertRaises(ValueError):
            wheel.parse_emotion_label(True, "item")

    def test_parse_original_annotations_and_join_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            anno = tmp / "IEMOCAP_full_release/Session1/dialog/EmoEvaluation"
            anno.mkdir(parents=True)
            entries = [
                ("Ses01F_impro01_F000", "neu", 0, 3, 2, 3),
                ("Ses01F_impro01_F001", "sad", 1, 1, 1, 3),
                ("Ses01F_impro01_F002", "fru", 2, 1, 4, 3),
                ("Ses01F_impro01_F003", "ang", 3, 1, 5, 4),
                ("Ses01F_impro01_F004", "hap", 4, 5, 3, 4),
                ("Ses01F_impro01_F005", "exc", 5, 5, 5, 4),
                ("Ses01F_impro01_F006", "hap", 4, 4, 3, 3),
                ("Ses01F_impro01_F007", "sad", 1, 1, 2, 2),
            ]
            with (anno / "Ses01F_impro01.txt").open("w") as handle:
                for i, (instance, category, idx, v, a, d) in enumerate(entries):
                    handle.write(f"[{i:.3f} - {i+1:.3f}]  {instance}  {category}  [{v:.2f}, {a:.2f}, {d:.2f}]\n")
            by_id, parse_stats = wheel.read_annotation_files(tmp / "IEMOCAP_full_release")
            self.assertEqual(len(by_id), 8)
            self.assertEqual(by_id[entries[2][0]]["activation"], 4)
            split = {"train": [], "dev": [], "test": []}
            for i, (instance, category, idx, v, a, d) in enumerate(entries):
                which = "train" if i < 6 else ("dev" if i == 6 else "test")
                label_value = category if i % 2 == 0 else idx
                split[which].append([instance, "speaker", "file", "audio", "text", label_value])
            split_path = tmp / "split.json"
            split_path.write_text(json.dumps(split), encoding="utf-8")
            output = tmp / "result/manifest.csv"
            wheel.prepare(SimpleNamespace(
                split_json=split_path, iemocap_root=tmp / "IEMOCAP_full_release",
                vad_csv=None, out=output, min_coverage=1.0))
            self.assertTrue(output.exists())
            self.assertEqual(len(wheel.read_manifest(output)), 8)
            self.assertEqual(wheel.load_json(tmp / "result/coverage.json")["missing_vad_count"], 0)
            audit = SimpleNamespace(
                manifest=output, split="dev", dims="VA", predictions=None,
                attempts_per_anchor=5, same_cross_draws=10, seed=7, top_examples=10,
                output=tmp / "audit")
            wheel.analyze(audit)
            summary = wheel.load_json(tmp / "audit/summary.json")
            self.assertEqual(summary["train_samples_with_vad"], 6)
            self.assertEqual(summary["study_samples_with_vad"], 1)
            self.assertEqual(summary["class_pair_count"], 10)
            self.assertTrue((tmp / "audit/class_pair_relationship.csv").is_file())


    def test_within_class_can_be_farther_than_cross_class(self):
        rows = [
            make_vad_row("a", "dev", "hap", 1, 1),
            make_vad_row("b", "dev", "hap", 5, 5),
            make_vad_row("c", "dev", "sad", 1, 1),
        ]
        out = wheel.same_vs_cross_class_audit(
            rows, ("valence", "activation"), seed=11, draws_per_relation=30)
        chosen = next(r for r in out if r["anchor_class"] == "hap" and
                      r["other_class"] == "sad")
        self.assertTrue(0 < chosen["cross_class_closer_than_same_rate"] < 1)

    def test_predictions_match_split_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pred.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle,
                                        fieldnames=["instance", "ground_truth_idx", "pred_idx"])
                writer.writeheader()
                writer.writerow({"instance": "x", "ground_truth_idx": 1, "pred_idx": 0})
            predictions = wheel.prediction_rows(path)
            self.assertEqual(predictions["x"], {"true": "sad", "pred": "neu"})


if __name__ == "__main__":
    unittest.main()
