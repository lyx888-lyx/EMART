import sys
from pathlib import Path

import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path("/code/EMART")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inference"))

from inference import EmotionInference
from dataloader.dataloader import (
    load_finetune_audios,
    set_finetune_dataloader,
)

torch.set_grad_enabled(False)

print("========== 加载作者权重 ==========", flush=True)

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

# 使用训练阶段的 DataLoader 设置
args = inf.args
args.load_pt = True
args.batch_size = 32
args.num_workers = 0
args.max_audio_len = 6

model.text_model.load_pt = True

_, _, test_files = load_finetune_audios(
    input_path=str(
        ROOT / "train_split/train_split_meld_repro"
    ),
    audio_path=str(ROOT / "audio"),
    dataset="meld7",
    fold_idx=1,
)

loader = set_finetune_dataloader(
    args,
    test_files,
    is_train=False,
)

print("测试样本:", len(test_files))
print("测试 batch:", len(loader))
print("multimodal_pooling:", model.multimodal_pooling)
print("text pooling_mode:", model.text_model.pooling_mode)

print("\n========== 开始批量评估 ==========", flush=True)

preds = []
truths = []

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

        pred = outputs.argmax(dim=-1)

        preds.extend(pred.cpu().tolist())
        truths.extend(y.tolist())

        if (i + 1) % 20 == 0:
            print(
                f"完成 {i+1}/{len(loader)} batches",
                flush=True
            )

acc = accuracy_score(truths, preds) * 100
wf1 = f1_score(
    truths, preds, average="weighted"
) * 100

print("\n========== 训练式评估结果 ==========")
print("样本数:", len(preds))
print(f"Accuracy: {acc:.2f}%")
print(f"Weighted F1: {wf1:.2f}%")

# 与之前的独立推理结果逐条比较
old_path = (
    ROOT / "test_results/mm_test_meld_wn_fixed/"
    "meld7_fold1_test_results.csv"
)

old = pd.read_csv(old_path)
old_pred = dict(
    zip(old["instance"], old["pred_idx"])
)

changed = []
for record, new_pred in zip(test_files, preds):
    uid = record[0]
    if uid in old_pred and int(old_pred[uid]) != new_pred:
        changed.append(uid)

print("\n========== 两套推理对照 ==========")
print("独立推理 ACC: 49.16%")
print(f"训练式评估 ACC: {acc:.2f}%")
print("预测不一致数量:", len(changed))
print("不一致示例:", changed[:15])

print("\n========== 完成 ==========")
