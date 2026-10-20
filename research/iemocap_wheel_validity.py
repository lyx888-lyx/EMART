#!/usr/bin/env python3
"""Audit EMART's fixed emotion-wheel geometry against IEMOCAP utterance V/A/D.

Standard library only. This is a descriptive, hypothesis-discovery experiment,
NOT evidence that the EMART wheel loss caused any observed classification error.
"""
import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ["neu", "sad", "fru", "ang", "hap", "exc"]
ANGLES = {
    "sad": 9 * math.pi / 8,
    "fru": 7 * math.pi / 8,
    "ang": 6 * math.pi / 8,
    "hap": math.pi / 8,
    "exc": 3 * math.pi / 8,
}
ANNOTATION_LINE = re.compile(
    r"^\s*\[\s*[\d.]+\s*-\s*[\d.]+\s*\]\s+"
    r"(Ses\d{2}[FM]_[A-Za-z0-9_]+_[FM]\d{3})\s+"
    r"([A-Za-z]+)\s+\[([^\]]+)\]"
)
NUMERIC = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
CSV_COLUMN_NAMES = {
    "instance": ("instance", "utterance_id", "utteranceid", "utt_id", "id", "speaker"),
    "valence": ("valence", "val", "v"),
    "activation": ("activation", "arousal", "act", "a"),
    "dominance": ("dominance", "dom", "d"),
}


def load_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, content):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")


def write_rows(path, rows, fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def angle_distance(first, second):
    a = abs(ANGLES[first] - ANGLES[second])
    return min(a, 2 * math.pi - a)


def distance(a, b, dims=("valence", "activation")):
    return math.sqrt(sum((float(a[d]) - float(b[d])) ** 2 for d in dims))


def mean(values):
    return sum(values) / len(values) if values else None


def midranks(values):
    sorted_values = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(sorted_values):
        end = index + 1
        while end < len(sorted_values) and sorted_values[end][1] == sorted_values[index][1]:
            end += 1
        rank = (index + 1 + end) / 2
        for pos in range(index, end):
            ranks[sorted_values[pos][0]] = rank
        index = end
    return ranks


def pearson(xs, ys):
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def read_annotation_files(root):
    root = Path(root)
    paths = sorted(root.glob("Session*/dialog/EmoEvaluation/*.txt"))
    if not paths:
        paths = sorted(root.rglob("EmoEvaluation/*.txt"))
    if not paths:
        raise FileNotFoundError(
            "Cannot find IEMOCAP dimensional annotations beneath "
            f"{root}. Expected Session*/dialog/EmoEvaluation/*.txt. "
            "If you have a separately exported VAD table, pass --vad-csv PATH."
        )
    records = {}
    examined = parsed = 0
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                examined += 1
                match = ANNOTATION_LINE.match(line)
                if not match:
                    continue
                sample_id, annotated_label, raw_scores = match.groups()
                numbers = NUMERIC.findall(raw_scores)
                if len(numbers) < 3:
                    continue
                values = list(map(float, numbers[:3]))
                if not all(math.isfinite(value) for value in values):
                    continue
                if sample_id in records:
                    raise ValueError(f"Duplicate annotation ID {sample_id} in {path}")
                records[sample_id] = {
                    "valence": values[0], "activation": values[1], "dominance": values[2],
                    "annotation_label": annotated_label.lower(), "source": str(path),
                }
                parsed += 1
    if not records:
        raise ValueError(f"No usable utterance-level VAD annotations in {len(paths)} files")
    return records, {"annotation_files": len(paths), "lines_scanned": examined,
                     "annotations_parsed": parsed, "annotation_mode": "original_txt"}


def read_vad_csv(path):
    rows = load_rows(path)
    if not rows:
        raise ValueError(f"Empty VAD CSV {path}")
    header = {column.strip().lower(): column for column in rows[0]}
    mapping = {}
    for target, choices in CSV_COLUMN_NAMES.items():
        original = next((header[key] for key in choices if key in header), None)
        if original is None:
            raise ValueError(f"CSV missing '{target}'. Accepted column names: {choices}")
        mapping[target] = original
    records = {}
    for n, row in enumerate(rows, 2):
        sample_id = row[mapping["instance"]].strip()
        if not sample_id:
            raise ValueError(f"Blank sample ID in CSV line {n}")
        if sample_id in records:
            raise ValueError(f"Duplicate VAD ID {sample_id} in CSV line {n}")
        rec = {dim: float(row[mapping[dim]]) for dim in ("valence", "activation", "dominance")}
        if not all(math.isfinite(x) for x in rec.values()):
            raise ValueError(f"Non-finite VAD in CSV line {n}")
        rec.update(annotation_label="", source=str(path))
        records[sample_id] = rec
    return records, {"annotation_files": 1, "annotations_parsed": len(records),
                     "annotation_mode": "csv"}


def split_records(path):
    raw = load_json(path)
    records, duplicates = {}, []
    counts = {}
    for split in ("train", "dev", "test"):
        samples = raw.get(split)
        if not isinstance(samples, list):
            raise ValueError(f"Split JSON must have list key '{split}': {path}")
        counts[split] = len(samples)
        for sample in samples:
            if not isinstance(sample, (list, tuple)) or len(sample) < 2:
                raise ValueError(f"Unexpected split row: {sample}")
            sample_id = str(sample[0])
            index = int(sample[-1])
            if index < 0 or index >= len(LABELS):
                raise ValueError(f"Unexpected IEMOCAP6 label index {index}: {sample_id}")
            if sample_id in records:
                duplicates.append(sample_id)
                continue
            records[sample_id] = {"instance": sample_id, "split": split,
                                  "emotion": LABELS[index], "label_index": index}
    if duplicates:
        raise ValueError(f"Duplicate utterance IDs across splits: {duplicates[:8]}")
    return records, counts


def prepare(args):
    from_split, counts = split_records(args.split_json)
    if args.vad_csv:
        vad, metadata = read_vad_csv(args.vad_csv)
    else:
        vad, metadata = read_annotation_files(args.iemocap_root)
    result, misses, mismatches, out_of_range = [], [], [], []
    matched = Counter()
    for key, row in from_split.items():
        if key not in vad:
            misses.append(row)
            continue
        score = vad[key]
        record = {**row, **score}
        if score["annotation_label"] and score["annotation_label"] != row["emotion"]:
            mismatches.append({"instance": key, "split_label": row["emotion"],
                               "annotation_label": score["annotation_label"]})
            # Exclude because label mapping is unverified for this utterance.
            continue
        if not all(1 <= score[dim] <= 5 for dim in ("valence", "activation", "dominance")):
            out_of_range.append(key)
        result.append(record)
        matched[row["split"]] += 1
    if not result:
        raise ValueError(
            "Zero matching IEMOCAP sample IDs. Inspect the split JSON and "
            "annotation IDs; do not calculate results on unrelated datasets."
        )
    fields = ["instance", "split", "emotion", "label_index", "valence",
              "activation", "dominance", "annotation_label", "source"]
    write_rows(args.out, result, fields)
    diagnostics = {
        **metadata, "split_source": str(args.split_json),
        "split_counts": counts, "vad_matched_by_split": dict(matched),
        "coverage_by_split": {split: matched[split] / count if count else None
                              for split, count in counts.items()},
        "missing_vad_count": len(misses),
        "annotation_label_mismatch_count_excluded": len(mismatches),
        "out_of_expected_1_to_5_count": len(out_of_range),
        "dimension_order": ["valence", "activation", "dominance"],
        "notes": [
            "Emotion-wheel comparisons exclude neutral because it has no paper-assigned angle.",
            "Records with disagreement between split class and annotated class are excluded.",
            "VAD scores are read directly from human annotation files; they are not model predictions."
        ]
    }
    write_json(Path(args.out).with_name("coverage.json"), diagnostics)
    write_rows(Path(args.out).with_name("missing_vad.csv"), misses,
               ["instance", "split", "emotion", "label_index"])
    write_rows(Path(args.out).with_name("label_mismatches.csv"), mismatches,
               ["instance", "split_label", "annotation_label"])
    print(json.dumps(diagnostics, indent=2))
    print(f"Manifest: {args.out}")
    if any(v is not None and v < args.min_coverage for v in diagnostics["coverage_by_split"].values()):
        raise SystemExit(
            f"WARNING: some splits have coverage below {args.min_coverage:.0%}. "
            "Inspect coverage.json and missing_vad.csv before interpreting results."
        )


def read_manifest(path):
    records = []
    seen = set()
    for row in load_rows(path):
        key = row["instance"]
        if key in seen:
            raise ValueError(f"Duplicate manifest ID {key}")
        seen.add(key)
        if row["emotion"] not in LABELS:
            raise ValueError(f"Unknown IEMOCAP label {row['emotion']}")
        if row["split"] not in ("train", "dev", "test"):
            raise ValueError(f"Unknown split {row['split']}")
        for dim in ("valence", "activation", "dominance"):
            row[dim] = float(row[dim])
            if not math.isfinite(row[dim]):
                raise ValueError(f"Invalid {dim} for {key}")
        records.append(row)
    return records


def centroids(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["emotion"]].append(row)
    output = {}
    for category, members in grouped.items():
        dimensions = ("valence", "activation", "dominance")
        center = {dim: mean([r[dim] for r in members]) for dim in dimensions}
        dispersion = {
            dim + "_sd": math.sqrt(mean([(r[dim] - center[dim]) ** 2 for r in members]))
            for dim in dimensions
        }
        output[category] = {"label": category, "n": len(members),
                            **center, **dispersion}
    return output


def prediction_rows(path):
    if path is None:
        return {}
    matched = {}
    for row in load_rows(path):
        key = (row.get("instance") or "").strip()
        if not key:
            raise ValueError(f"Missing instance ID in prediction CSV: {path}")
        if key in matched:
            raise ValueError(f"Duplicate prediction instance: {key}")
        true_label = row.get("ground_truth_label") or row.get("ground_truth")
        pred_label = row.get("pred_label") or row.get("prediction")
        if not true_label and row.get("ground_truth_idx", "") != "":
            true_label = LABELS[int(row["ground_truth_idx"])]
        if not pred_label and row.get("pred_idx", "") != "":
            pred_label = LABELS[int(row["pred_idx"])]
        matched[key] = {"true": true_label, "pred": pred_label}
    return matched


def class_pair_relationship(train_centroids, dims):
    categories = [x for x in LABELS if x in ANGLES and x in train_centroids]
    rows = []
    for i, a in enumerate(categories):
        for b in categories[i + 1:]:
            rows.append({
                "class_a": a, "class_b": b,
                "wheel_angle_degrees": math.degrees(angle_distance(a, b)),
                "train_centroid_distance": distance(train_centroids[a], train_centroids[b], dims),
            })
    corr = pearson(midranks([r["wheel_angle_degrees"] for r in rows]),
                   midranks([r["train_centroid_distance"] for r in rows]))
    return rows, corr


def sample_prior_inversion(anchor, other_a, other_b, dims):
    angle_a = angle_distance(anchor["emotion"], other_a["emotion"])
    angle_b = angle_distance(anchor["emotion"], other_b["emotion"])
    if math.isclose(angle_a, angle_b, abs_tol=1e-10):
        return None
    actual_a = distance(anchor, other_a, dims)
    actual_b = distance(anchor, other_b, dims)
    if math.isclose(actual_a, actual_b, abs_tol=1e-10):
        return None
    return int((angle_a - angle_b) * (actual_a - actual_b) < 0)


def instance_prior_inversion(row, train_centroids, dims):
    category = row["emotion"]
    if category not in ANGLES:
        return None, 0
    others = [name for name in ANGLES if name != category and name in train_centroids]
    n, violations = 0, 0
    for index, a in enumerate(others):
        for b in others[index + 1:]:
            prior_a, prior_b = angle_distance(category, a), angle_distance(category, b)
            if math.isclose(prior_a, prior_b, abs_tol=1e-10):
                continue
            observed_a = distance(row, train_centroids[a], dims)
            observed_b = distance(row, train_centroids[b], dims)
            if math.isclose(observed_a, observed_b, abs_tol=1e-10):
                continue
            n += 1
            violations += ((prior_a - prior_b) * (observed_a - observed_b) < 0)
    return (violations / n if n else None), n


def triplet_audit(rows, dims, seed, attempts_per_anchor):
    rng = random.Random(seed)
    class_rows = defaultdict(list)
    for row in rows:
        if row["emotion"] in ANGLES:
            class_rows[row["emotion"]].append(row)
    examined = inverted = 0
    by_anchor = {}
    for anchor in rows:
        label = anchor["emotion"]
        if label not in ANGLES:
            continue
        alternatives = [c for c in ANGLES if c != label and class_rows.get(c)]
        n, errors = 0, 0
        if len(alternatives) < 2:
            continue
        for _ in range(attempts_per_anchor):
            a, b = rng.sample(alternatives, 2)
            p, q = rng.choice(class_rows[a]), rng.choice(class_rows[b])
            inversion = sample_prior_inversion(anchor, p, q, dims)
            if inversion is None:
                continue
            n += 1
            errors += inversion
        if n:
            by_anchor[anchor["instance"]] = {"instance": anchor["instance"],
                                             "emotion": label, "n_triplets": n,
                                             "n_inversions": errors,
                                             "triplet_inversion_rate": errors / n}
            examined += n
            inverted += errors
    return list(by_anchor.values()), {"triplets_evaluated": examined, "triplets_inverted": inverted,
                                     "triplet_inversion_rate": inverted / examined if examined else None,
                                     "sampling": "anchors enumerated once; comparison categories uniformly sampled;" 
                                                 " instances sampled within category, fixed RNG"}


def same_vs_cross_class_audit(rows, dims, seed, draws_per_relation):
    """How often is another category closer in VA(D) than another member of the same class?

    Draw same- and different-class samples for each anchor category and each
    other category independently, to reduce confounding by label prevalence.
    """
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for row in rows:
        if row["emotion"] in ANGLES:
            grouped[row["emotion"]].append(row)
    output = []
    for own_class in ANGLES:
        own = grouped[own_class]
        if len(own) < 2:
            continue
        for other_class in ANGLES:
            if other_class == own_class or not grouped[other_class]:
                continue
            overlaps = 0
            within = []
            across = []
            for _ in range(draws_per_relation):
                anchor, positive = rng.sample(own, 2)
                negative = rng.choice(grouped[other_class])
                d_within = distance(anchor, positive, dims)
                d_across = distance(anchor, negative, dims)
                within.append(d_within)
                across.append(d_across)
                overlaps += int(d_within > d_across)
            output.append({
                "anchor_class": own_class, "other_class": other_class,
                "angle_distance_degrees": math.degrees(angle_distance(own_class, other_class)),
                "n_draws": draws_per_relation,
                "cross_class_closer_than_same_rate": overlaps / draws_per_relation,
                "mean_same_class_distance": mean(within),
                "mean_cross_class_distance": mean(across),
            })
    return output


def mismatch_bin(value):
    if value is None:
        return "undefined"
    if value == 0:
        return "0"
    if value <= 1 / 3 + 1e-12:
        return "(0, 1/3]"
    if value <= 2 / 3 + 1e-12:
        return "(1/3, 2/3]"
    return "(2/3, 1]"


def analyze(args):
    all_rows = read_manifest(args.manifest)
    training = [row for row in all_rows if row["split"] == "train"]
    study = [row for row in all_rows if row["split"] == args.split]
    if not training or not study:
        raise ValueError(f"Need train and {args.split} manifest rows with nonzero coverage.")
    training_centroids = centroids(training)
    if not all(cls in training_centroids for cls in ANGLES):
        raise ValueError("Training VAD must cover all 5 non-neutral categories.")
    dims = ("valence", "activation") if args.dims == "VA" else ("valence", "activation", "dominance")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    train_centroid_rows = [training_centroids[cls] for cls in LABELS if cls in training_centroids]
    write_rows(output / "train_class_centroids.csv", train_centroid_rows,
               ["label", "n", "valence", "activation", "dominance",
                "valence_sd", "activation_sd", "dominance_sd"])
    geometry_pairs, rho = class_pair_relationship(training_centroids, dims)
    write_rows(output / "class_pair_relationship.csv", geometry_pairs,
               ["class_a", "class_b", "wheel_angle_degrees", "train_centroid_distance"])
    preds = prediction_rows(args.predictions)
    joined = []
    mismatched_labels = []
    for row in study:
        mis, pairs = instance_prior_inversion(row, training_centroids, dims)
        prediction = preds.get(row["instance"])
        if prediction and prediction["true"] != row["emotion"]:
            mismatched_labels.append(row["instance"])
            continue
        rec = {k: row[k] for k in ("instance", "split", "emotion", "valence",
                                  "activation", "dominance")}
        rec.update({
            "prior_inversion_rate": mis,
            "comparable_class_pairs": pairs,
            "own_train_class_centroid_distance": distance(row, training_centroids[row["emotion"]], dims),
            "mismatch_bin": mismatch_bin(mis),
            "pred": prediction["pred"] if prediction else "",
            "correct": int(prediction["pred"] == row["emotion"]) if prediction else "",
        })
        joined.append(rec)
    if mismatched_labels:
        raise ValueError(
            "Ground-truth mismatch between predictions and VAD manifest. "
            f"Examples: {mismatched_labels[:5]}. Verify same fold and preprocessing!"
        )
    row_fields = ["instance", "split", "emotion", "valence", "activation", "dominance",
                  "prior_inversion_rate", "comparable_class_pairs",
                  "own_train_class_centroid_distance", "mismatch_bin", "pred", "correct"]
    write_rows(output / "per_instance.csv", joined, row_fields)
    empirical_rows = [row for row in study if row["emotion"] in ANGLES]
    triplets, triplet_stats = triplet_audit(empirical_rows, dims, args.seed, args.attempts_per_anchor)
    write_rows(output / "triplet_per_anchor.csv", triplets,
               ["instance", "emotion", "n_triplets", "n_inversions", "triplet_inversion_rate"])

    same_cross = same_vs_cross_class_audit(study, dims, args.seed + 1, args.same_cross_draws)
    write_rows(output / "same_vs_cross_class.csv", same_cross,
               ["anchor_class", "other_class", "angle_distance_degrees", "n_draws",
                "cross_class_closer_than_same_rate", "mean_same_class_distance",
                "mean_cross_class_distance"])

    grouped = defaultdict(list)
    for row in joined:
        grouped[(row["emotion"], row["mismatch_bin"])].append(row)
    bins = []
    for (emotion, bin_name), members in sorted(grouped.items()):
        with_prediction = [r for r in members if r["correct"] != ""]
        bins.append({
            "emotion": emotion, "mismatch_bin": bin_name, "n": len(members),
            "mean_prior_inversion": mean([r["prior_inversion_rate"] for r in members
                                           if r["prior_inversion_rate"] is not None]),
            "n_with_prediction": len(with_prediction),
            "error_rate": mean([1 - r["correct"] for r in with_prediction]),
            "mean_within_class_distance": mean([r["own_train_class_centroid_distance"]
                                                for r in members]),
        })
    write_rows(output / "error_by_class_and_mismatch.csv", bins,
               ["emotion", "mismatch_bin", "n", "mean_prior_inversion",
                "n_with_prediction", "error_rate", "mean_within_class_distance"])
    paired_pred = [r for r in joined if r["correct"] != "" and r["emotion"] != "neu"]
    mismatch_wrong = [r["prior_inversion_rate"] for r in paired_pred
                      if r["correct"] == 0 and r["prior_inversion_rate"] is not None]
    mismatch_correct = [r["prior_inversion_rate"] for r in paired_pred
                        if r["correct"] == 1 and r["prior_inversion_rate"] is not None]
    examples = sorted([r for r in paired_pred if r["correct"] == 0 and
                       r["prior_inversion_rate"] is not None],
                      key=lambda r: (-r["prior_inversion_rate"], r["instance"]))
    write_rows(output / "high_mismatch_errors.csv", examples[:args.top_examples], row_fields)
    summary = {
        "study_split": args.split,
        "dimensions": args.dims,
        "train_samples_with_vad": len(training),
        "study_samples_with_vad": len(study),
        "study_samples_non_neutral": len(empirical_rows),
        "study_prediction_matches": len([r for r in joined if r["correct"] != ""]),
        "prediction_csv_rows": len(preds),
        "class_pair_count": len(geometry_pairs),
        "class_pair_spearman_rho_descriptive": rho,
        "triplet": triplet_stats,
        "same_vs_cross_class_relation_count": len(same_cross),
        "mean_instance_prior_inversion": mean([r["prior_inversion_rate"] for r in joined
                                              if r["prior_inversion_rate"] is not None]),
        "mean_prior_inversion_wrong_predictions": mean(mismatch_wrong),
        "mean_prior_inversion_correct_predictions": mean(mismatch_correct),
        "warnings": [
            "This audit measures agreement with independent human VA(D) ratings, NOT causal harm from L_emo.",
            "No neutral angle: exclude neutral from wheel-angle comparisons but retain its descriptive rows.",
            "Class-centroid distances use train split only; evaluate mismatch on held-out split.",
            "Triplets reuse utterances; naive iid confidence intervals would be invalid.",
            "Testing an effect of the wheel loss requires a matched no-L_emo training intervention.",
            "VA dimensions and 1-5 annotation values have measurement limits; do not equate VA distance with ground-truth cosine embedding angle.",
            "If examining test predictions, do not select hyperparameters based on these test results."
        ],
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Detailed evidence files: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    first = commands.add_parser("prepare", help="Join continuous VAD annotations to current EMART split")
    first.add_argument("--iemocap-root", type=Path, default=Path("../dataset/IEMOCAP_full_release"))
    first.add_argument("--vad-csv", type=Path, help="Optional external utterance-ID keyed VAD CSV")
    first.add_argument("--split-json", type=Path,
                       default=Path("train_split/train_split_iemocap/iemocap6_fold1.json"))
    first.add_argument("--out", type=Path, default=Path("research_outputs/iemocap_wheel/manifest.csv"))
    first.add_argument("--min-coverage", type=float, default=0.8,
                       help="Exit nonzero when any split has lower VAD coverage; default .8")
    first.set_defaults(func=prepare)

    second = commands.add_parser("analyze", help="Compare original wheel with train-derived VA relationships")
    second.add_argument("--manifest", type=Path,
                        default=Path("research_outputs/iemocap_wheel/manifest.csv"))
    second.add_argument("--split", choices=["dev", "test"], default="dev")
    second.add_argument("--dims", choices=["VA", "VAD"], default="VA",
                        help="Primary hypothesis is planar emotion wheel vs valence/activation")
    second.add_argument("--predictions", type=Path, default=None,
                        help="Optional CSV with instance/ground_truth/prediction columns; must match chosen split")
    second.add_argument("--attempts-per-anchor", type=int, default=30)
    second.add_argument("--seed", type=int, default=42)
    second.add_argument("--same-cross-draws", type=int, default=1000)
    second.add_argument("--top-examples", type=int, default=60)
    second.add_argument("--output", type=Path, default=Path("research_outputs/iemocap_wheel/dev_audit"))
    second.set_defaults(func=analyze)

    args = parser.parse_args()
    if args.command == "analyze" and args.attempts_per_anchor < 1:
        parser.error("--attempts-per-anchor must be positive")
    if args.command == "analyze" and args.same_cross_draws < 1:
        parser.error("--same-cross-draws must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
