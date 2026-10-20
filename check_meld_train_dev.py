import sys
import random
from pathlib import Path
from collections import Counter

import torch
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path("/code/EMART")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inference"))

from inference import EmotionInference
from dataloader.dataloader import (
    load_finetune_audios,
    set_finetune_dataloader,
)

random.seed(8)
torch.manual_seed(8)

print("===== 加载作者 MELD checkpoint =====", flush=True)

inf = EmotionInference(
    model_dir=str(
        ROOT / "finetune/meld7/pretrained_wn_compat"
    ),
    fold_idx=1,
    device="cuda",
    dataset="meld7",
    modal="multimodal",
)

model = inf.model
model.eval()
model.text_model.load_pt = True

args = inf.args
args.load_pt = True
args.batch_size = 32
args.num_workers = 0
args.max_audio_len = 6

train_files, dev_files, test_files = load_finetune_audios(
    input_path=str(
        ROOT / "train_split/train_split_meld_repro"
    ),
    audio_path=str(ROOT / "audio"),
    dataset="meld7",
    fold_idx=1,
)

print("\n===== 数据集大小 =====")
print("Train:", len(train_files))
print("Dev:", len(dev_files))
print("Test:", len(test_files))

# 随机抽取训练样本，避免顺序采样偏差
sampled_train = random.sample(
    train_files, min(1500, len(train_files))
)

labels = [
    "neutral", "sadness", "anger",
    "joy", "surprise", "fear", "disgust"
]

def evaluate(name, records):
    loader = set_finetune_dataloader(
        args, records, is_train=False
    )

    preds = []
    truths = []

    print(f"\n===== {name} 评估 =====", flush=True)

    with torch.inference_mode():
        for i, batch in enumerate(loader):
            x, x_text, _, y, lengths = batch

            x = x.to(inf.device)
            lengths = lengths.to(inf.device)

            outputs, _, _, _ = model(
                audio_input=x,
                text_input=x_text,
                length=lengths,
            )

            p = outputs.argmax(dim=-1)

            preds.extend(p.cpu().tolist())
            truths.extend(y.tolist())

            if (i + 1) % 20 == 0:
                print(
                    f"{name}: {i+1}/{len(loader)} batches",
                    flush=True
                )

    acc = accuracy_score(truths, preds) * 100
    wf1 = f1_score(
        truths, preds, average="weighted",
        zero_division=0
    ) * 100
    macro = f1_score(
        truths, preds, average="macro",
        zero_division=0
    ) * 100

    counts = Counter(truths)
    majority = max(counts.values()) / len(truths) * 100

    print(f"\n{name} 样本数: {len(truths)}")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Weighted F1: {wf1:.2f}%")
    print(f"Macro F1: {macro:.2f}%")
    print(f"多数类基线: {majority:.2f}%")

    print("\n各类别召回率:")
    for j, label in enumerate(labels):
        total = sum(v == j for v in truths)
        correct = sum(
            t == j and p == j
            for t, p in zip(truths, preds)
        )
        recall = 100 * correct / total if total else 0
        print(
            f"{label:10s}: {recall:6.2f}% "
            f"({correct}/{total})"
        )

    return acc, wf1

train_acc, train_f1 = evaluate(
    "TRAIN (1500随机样本)", sampled_train
)

dev_acc, dev_f1 = evaluate(
    "DEV (全部样本)", dev_files
)

print("\n========== 最终对照 ==========")
print(f"Train: ACC={train_acc:.2f}% WF1={train_f1:.2f}%")
print(f"Dev:   ACC={dev_acc:.2f}% WF1={dev_f1:.2f}%")
print("Test:  ACC=49.16% WF1=46.73% (此前结果)")
print("论文 Test: ACC=66.70% WF1=64.71%")
