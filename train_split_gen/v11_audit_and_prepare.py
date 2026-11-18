#!/usr/bin/env python3
"""V11: verify the actual cached IEMOCAP tokens, then prepare matched TRAIN/DEV inputs.

Run from the EMART root. Never preprocess official TEST or train a model.
The legacy arm reproduces the source's full-utterance identity comparison and
token registration order. Alignment changes token IDs only. Role correction
changes SELF/OTHER only, with exactly the same selected history and token budget.
Do not use regenerated "legacy" as a reproduction unless local tokens match.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import RobertaTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODES = ("legacy", "aligned", "role_fixed")
EMOTIONS = {"neu", "sad", "fru", "ang", "hap", "exc"}
MAX_LEN = 256


def identity(utterance_id):
    match = re.fullmatch(r"(Ses0[1-5][FM]_.+)_([FM])(\d+)", str(utterance_id))
    if not match:
        raise ValueError(f"Not an IEMOCAP utterance ID: {utterance_id}")
    return match.group(1), match.group(2)


def tokenizers():
    legacy = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaTokenizer.from_pretrained("roberta-base")
    legacy.add_special_tokens({"additional_special_tokens":
                               ["[Current]", "<SELF>", "<OTHER>"]})
    model.add_special_tokens({"additional_special_tokens":
                              ["[Previous]", "[Current]", "<SELF>", "<OTHER>"]})
    return legacy, model


def make_contexts(df, legacy):
    required = {"Dialogue_ID", "Start_Time", "Speaker", "Utterance", "Emotion"}
    if not required.issubset(df.columns):
        raise ValueError("Missing CSV columns: " + str(required-set(df.columns)))
    if df[list(required)].isna().any().any():
        raise ValueError("Missing dialogue, utterance, timestamp, or emotion in CSV")
    if df["Speaker"].duplicated().any():
        raise ValueError("Duplicate full utterance IDs in CSV")
    df = df.sort_values(["Dialogue_ID", "Start_Time"], ascending=[True, True]).reset_index(drop=True)
    records = {}
    for dialogue, group in df.groupby("Dialogue_ID", sort=False):
        rows = list(group.to_dict("records"))
        parsed = [identity(r["Speaker"]) for r in rows]
        if len({d for d, _ in parsed}) != 1:
            raise ValueError(f"Dialogue ID grouping mixes recordings: {dialogue}")
        for i, row in enumerate(rows):
            current = f"[Current] <SELF> {row['Utterance']}"
            old = current
            selected = []
            # Match the original preprocessing's most-recent-first selection
            # and stop at the first segment exceeding the length budget.
            for j in range(i-1, -1, -1):
                segment = f"<OTHER> {rows[j]['Utterance']}"
                candidate = f"{segment} </s></s> {old}"
                if len(legacy.encode(candidate, add_special_tokens=True)) > MAX_LEN:
                    break
                old = candidate
                selected.append(j)
            fixed = current
            for j in selected:
                role = "<SELF>" if parsed[j][1] == parsed[i][1] else "<OTHER>"
                fixed = f"{role} {rows[j]['Utterance']} </s></s> {fixed}"
            records[str(row["Speaker"])] = {
                "legacy": old, "aligned": old, "role_fixed": fixed,
                "historical_self": sum(parsed[j][1] == parsed[i][1] for j in selected),
                "label": str(row["Emotion"]),
                "dialogue": parsed[i][0],
            }
    return records


def encode(tok, text):
    out = tok(text, padding="max_length", truncation=True,
              max_length=MAX_LEN, return_tensors="pt")
    return {k: out[k].squeeze(0).cpu() for k in ("input_ids", "attention_mask")}


def equal_tokens(a, b):
    return all(torch.equal(torch.as_tensor(a[k]), torch.as_tensor(b[k]))
               for k in ("input_ids", "attention_mask"))


def resolve_training_path(value):
    p = Path(value)
    # Original finetune.py runs from experiment/ and uses paths verbatim.
    return p.resolve() if p.is_absolute() else (ROOT/"experiment"/p).resolve()


def find_csv_root(user_path):
    if user_path is not None:
        return user_path.resolve()
    candidates = [p/"dataset/IEMOCAP_full_release/telme_data"
                  for p in (ROOT, ROOT.parent)]
    found = [p for p in candidates if (p/"IEMOCAP_train.csv").is_file()
             and (p/"IEMOCAP_dev.csv").is_file()]
    if len(found) != 1:
        raise ValueError("Specify --csv-root (the directory with IEMOCAP_train.csv and IEMOCAP_dev.csv)")
    return found[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-json", type=Path, default=ROOT/"train_split/train_split_iemocap/iemocap6_fold1.json")
    p.add_argument("--csv-root", type=Path)
    p.add_argument("--output", type=Path, default=ROOT/"results/v11_roles")
    args = p.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}; use --output with a new directory")
    csv_root = find_csv_root(args.csv_root)
    raw = json.loads(args.split_json.read_text(encoding="utf-8"))
    legacy, model = tokenizers()
    special = ["[Current]", "<SELF>", "<OTHER>"]
    ids = {name: {t: int(tok.convert_tokens_to_ids(t)) for t in special}
           for name,tok in (("preprocess",legacy),("model",model))}
    print("V11 special token IDs:", json.dumps(ids), flush=True)
    manifests = {mode: {"train": [], "dev": []} for mode in MODES}
    caches = {mode: {} for mode in MODES}
    summary = {}
    dialogues = {}
    for split in ("train","dev"):
        rows = [row for row in raw[split] if row[-1] in EMOTIONS]
        if not rows or len({r[0] for r in rows}) != len(rows):
            raise ValueError(f"{split}: empty or duplicate utterances")
        csv_path = csv_root/f"IEMOCAP_{split}.csv"
        contexts = make_contexts(pd.read_csv(csv_path), legacy)
        dialogues[split] = {r["dialogue"] for r in contexts.values()}
        pt_paths = {resolve_training_path(r[4]) for r in rows}
        if len(pt_paths) != 1:
            raise ValueError(f"{split}: original loader requires one cached PT per split")
        saved_path = next(iter(pt_paths))
        if not saved_path.is_file():
            raise FileNotFoundError(f"Existing preprocessed data missing: {saved_path}")
        saved = torch.load(saved_path, map_location="cpu", weights_only=True)
        matches = {m:0 for m in MODES}
        history_self_turns = history_self_segments = role_changes = old_has_current = 0
        for mode in MODES:
            caches[mode][split] = {}
        for row in rows:
            uid = str(row[0])
            if uid not in contexts or uid not in saved:
                raise ValueError(f"{split}: ID absent from raw CSV or existing PT: {uid}")
            record = contexts[uid]
            if record["label"] != row[-1]:
                raise ValueError(f"{split}: label mismatch for {uid}")
            audio_path = resolve_training_path(row[3])
            if not audio_path.is_file():
                raise FileNotFoundError(f"Audio missing: {audio_path}")
            variants = {m: encode(legacy if m=="legacy" else model, record[m])
                        for m in MODES}
            if not torch.equal(variants["aligned"]["attention_mask"],
                               variants["role_fixed"]["attention_mask"]):
                raise RuntimeError("Role correction unexpectedly changes token budget")
            old_has_current += int(bool(((torch.as_tensor(saved[uid]["input_ids"]) ==
                                         ids["model"]["[Current]"]) &
                                        torch.as_tensor(saved[uid]["attention_mask"]).bool()).any()))
            history_self_turns += int(record["historical_self"]>0)
            history_self_segments += record["historical_self"]
            role_changes += int(not equal_tokens(variants["aligned"], variants["role_fixed"]))
            for mode in MODES:
                matches[mode] += int(equal_tokens(variants[mode], saved[uid]))
                variants[mode]["label"] = row[-1]
                caches[mode][split][uid] = variants[mode]
                new_row = list(row)
                new_row[3] = str(audio_path)
                new_row[4] = str(out/mode/f"{split}.pt")
                manifests[mode][split].append(new_row)
        summary[split] = dict(utterances=len(rows), dialogues=len(dialogues[split]),
                              existing_exact_matches=matches,
                              turns_with_true_self_history=history_self_turns,
                              self_history_segments=history_self_segments,
                              aligned_vs_fixed_changed_turns=role_changes,
                              existing_has_model_current_id=old_has_current,
                              existing_pt=str(saved_path),
                              csv_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest())
        print(split, json.dumps(summary[split], ensure_ascii=False), flush=True)
    if dialogues["train"] & dialogues["dev"]:
        raise RuntimeError("TRAIN/DEV RAW CSV dialogue overlap; do not train")
    matched = [m for m in MODES if all(s["existing_exact_matches"][m] ==
                                     s["utterances"] for s in summary.values())]
    # A positive count of the model's current ID alone does not prove
    # correct semantics: the legacy SELF token may use that same ID.
    if "role_fixed" in matched:
        arms, decision = [], "already_role_fixed"
    elif "aligned" in matched:
        arms, decision = ["aligned","role_fixed"], "role_fix_only"
    elif "legacy" in matched:
        arms, decision = ["legacy","aligned","role_fixed"], "token_and_role_fix"
    else:
        arms, decision = [], "local_cache_not_reproduced_stop"
    if not sum(s["aligned_vs_fixed_changed_turns"] for s in summary.values()):
        arms, decision = [], "no_actual_role_token_difference"
    out.mkdir(parents=True)
    report = dict(token_ids=ids, splits=summary, existing_matches=matched,
                  training_arms=arms, decision=decision,
                  official_test_preprocessed_or_scored=False)
    (out/"audit.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    for mode in arms:
        folder = out/mode
        folder.mkdir()
        for split in ("train","dev"):
            torch.save(caches[mode][split], folder/f"{split}.pt")
        # Only TRAIN/DEV rows; this manifest cannot drive TEST inference.
        (folder/"iemocap6_fold1.json").write_text(
            json.dumps(manifests[mode],ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("V11 AUDIT DECISION:", decision)
    print("TRAINING ARMS:", arms)
    print("Saved:", out/"audit.json")
    print("No model training, no official TEST.")


if __name__=="__main__":
    main()
