#!/usr/bin/env python3
"""Reproducible post-hoc error analysis of EMART prediction CSVs (no GPU needed)."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

LABELS = {
    "iemocap6": ["neu", "sad", "fru", "ang", "hap", "exc"],
    "meld7": ["neutral", "sadness", "anger", "joy", "surprise", "fear", "disgust"],
}


def read_rows(path, dataset):
    labels = LABELS[dataset]
    records, skipped = {}, []
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            sample_id = str(row.get("instance") or "").strip()
            if not sample_id or sample_id in records:
                raise ValueError(f"{path}:{line_number}: missing or duplicate instance {sample_id!r}")
            truth = (row.get("ground_truth_label") or row.get("ground_truth") or "").strip()
            pred = (row.get("pred_label") or row.get("prediction") or "").strip()
            if not truth and row.get("ground_truth_idx", "") != "":
                truth = labels[int(row["ground_truth_idx"])]
            if not pred and row.get("pred_idx", "") != "":
                index = int(row["pred_idx"])
                pred = labels[index] if 0 <= index < len(labels) else ""
            if truth not in labels:
                raise ValueError(f"{path}:{line_number}: unknown ground truth {truth!r}")
            if pred not in labels:
                skipped.append({"instance": sample_id, "reason": f"invalid prediction {pred!r}"})
                continue
            # Existing EMART inference CSVs have no probability columns; don't invent them.
            records[sample_id] = {"instance": sample_id, "true": truth, "pred": pred}
    if not records:
        raise ValueError(f"No valid predictions in {path}")
    return records, skipped


def calculate(records, labels):
    rows = list(records.values())
    n = len(rows)
    matrix = {(a, b): 0 for a in labels for b in labels}
    for row in rows:
        matrix[row["true"], row["pred"]] += 1
    correct = sum(matrix[c, c] for c in labels)
    per_class = []
    for c in labels:
        tp = matrix[c, c]
        support = sum(matrix[c, p] for p in labels)
        predicted = sum(matrix[t, c] for t in labels)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(dict(label=c, support=support, predicted=predicted, tp=tp,
                              precision=precision, recall=recall, f1=f1))
    neutral = labels[0]
    non_neutral = [r for r in rows if r["true"] != neutral]
    false_neutral = [r for r in non_neutral if r["pred"] == neutral]
    summary = {
        "n_valid": n,
        "accuracy": correct / n,
        "macro_f1": sum(c["f1"] for c in per_class) / len(labels),
        "weighted_f1": sum(c["f1"] * c["support"] for c in per_class) / n,
        "uar": sum(c["recall"] for c in per_class) / len(labels),
        "false_neutral_rate_given_non_neutral": len(false_neutral) / len(non_neutral) if non_neutral else None,
        "false_neutral_count": len(false_neutral),
        "non_neutral_count": len(non_neutral),
    }
    return summary, per_class, matrix, false_neutral


def write_csv(path, columns, rows):
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_analysis(records, labels, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, per_class, matrix, false_neutral = calculate(records, labels)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(out_dir / "per_class.csv",
              ["label", "support", "predicted", "tp", "precision", "recall", "f1"], per_class)
    write_csv(out_dir / "confusion_counts.csv", ["true"] + labels,
              [dict(true=t, **{p: matrix[t, p] for p in labels}) for t in labels])
    write_csv(out_dir / "confusion_row_normalized.csv", ["true"] + labels,
              [dict(true=t, **{p: matrix[t, p] / max(1, sum(matrix[t, q] for q in labels))
                                   for p in labels}) for t in labels])
    write_csv(out_dir / "error_flows.csv", ["true", "pred", "count", "share_of_true"],
              [dict(true=t, pred=p, count=matrix[t, p],
                    share_of_true=matrix[t, p] / max(1, sum(matrix[t, q] for q in labels)))
               for t in labels for p in labels if t != p and matrix[t, p] > 0])
    write_csv(out_dir / "false_neutral.csv", ["instance", "true", "pred"], false_neutral)
    return summary


def paired_comparison(base, candidate, labels, out_dir):
    shared = sorted(set(base) & set(candidate))
    if not shared:
        raise ValueError("No matching instance IDs between baseline and candidate")
    if any(base[k]["true"] != candidate[k]["true"] for k in shared):
        raise ValueError("Ground-truth mismatch for matched instance IDs; likely different data splits")
    paired = []
    by_class = {}
    for k in shared:
        t, b, c = base[k]["true"], base[k]["pred"], candidate[k]["pred"]
        paired.append(dict(instance=k, true=t, base_pred=b, candidate_pred=c,
                           base_correct=int(t == b), candidate_correct=int(t == c)))
    for label in labels:
        rr = [r for r in paired if r["true"] == label]
        by_class[label] = {
            "n": len(rr),
            "base_correct": sum(r["base_correct"] for r in rr),
            "candidate_correct": sum(r["candidate_correct"] for r in rr),
        }
    b_wrong_c_right = sum(r["candidate_correct"] and not r["base_correct"] for r in paired)
    b_right_c_wrong = sum(r["base_correct"] and not r["candidate_correct"] for r in paired)
    paired_summary = {
        "n_shared": len(shared),
        "n_only_baseline": len(base) - len(shared),
        "n_only_candidate": len(candidate) - len(shared),
        "baseline_wrong_candidate_right": b_wrong_c_right,
        "baseline_right_candidate_wrong": b_right_c_wrong,
        "paired_accuracy_delta": (b_wrong_c_right - b_right_c_wrong) / len(shared),
        "by_true_class": by_class,
    }
    (out_dir / "paired_summary.json").write_text(json.dumps(paired_summary, indent=2), encoding="utf-8")
    write_csv(out_dir / "paired_predictions.csv",
              ["instance", "true", "base_pred", "candidate_pred", "base_correct", "candidate_correct"], paired)
    return paired_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=LABELS)
    parser.add_argument("--predictions", required=True, type=Path, help="Baseline inference CSV")
    parser.add_argument("--compare", type=Path, help="Optional second run's inference CSV")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    base, skipped = read_rows(args.predictions, args.dataset)
    summary = save_analysis(base, LABELS[args.dataset], args.output)
    summary["n_invalid_predictions_excluded"] = len(skipped)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if args.compare:
        candidate, skipped_other = read_rows(args.compare, args.dataset)
        comparison = paired_comparison(base, candidate, LABELS[args.dataset], args.output)
        comparison["n_invalid_candidate_predictions_excluded"] = len(skipped_other)
        print("Paired comparison:", json.dumps(comparison, indent=2))
    print("Saved to", args.output)


if __name__ == "__main__":
    main()
