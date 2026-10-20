import csv
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm
from inference import EmotionInference

ROOT = Path(__file__).resolve().parents[1]
INFERENCE_DIR = ROOT / "inference"
CKPT = ROOT / "finetune/iemocap6/multimodal/pretrained/iemocap_fold_1.pt"
SPLIT = ROOT / "train_split/train_split_iemocap/iemocap6_fold1.json"
OUT = ROOT / "test_results/iemocap6_full_checkpoint/predictions.csv"
LABELS = ["neu", "sad", "fru", "ang", "hap", "exc"]


def load_complete_weights(self):
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    original_count = len(sd)

    # 对齐作者 checkpoint 中 RoBERTa 的词表大小
    emb_key = "text_model.semantic_model.embeddings.word_embeddings.weight"
    vocab_size = sd[emb_key].shape[0]

    emb = self.model.text_model.semantic_model.embeddings.word_embeddings

    if emb.num_embeddings != vocab_size:
        self.model.text_model.semantic_model.resize_token_embeddings(
            vocab_size
        )

    # 适配 PyTorch 1.13 weight_norm 参数命名
    prefix = "audio_model.backbone_model.encoder.pos_conv_embed.conv"

    for source, target in (
        (
            f"{prefix}.parametrizations.weight.original0",
            f"{prefix}.weight_g"
        ),
        (
            f"{prefix}.parametrizations.weight.original1",
            f"{prefix}.weight_v"
        ),
    ):
        expected = self.model.state_dict()

        if source not in sd or target not in expected or target in sd:
            raise RuntimeError(
                f"无法安全映射权重: {source} -> {target}"
            )

        if sd[source].shape != expected[target].shape:
            raise RuntimeError(f"权重形状不一致: {source}")

        sd[target] = sd.pop(source)

    # 处理 RoBERTa 非训练位置索引缓冲区
    position_key = "text_model.semantic_model.embeddings.position_ids"

    if position_key in sd and position_key not in self.model.state_dict():
        saved = sd[position_key].cpu()

        expected = torch.arange(
            saved.shape[-1], dtype=saved.dtype
        ).unsqueeze(0)

        if saved.shape != expected.shape or not torch.equal(saved, expected):
            raise RuntimeError("position_ids 异常，停止加载")

        del sd[position_key]

    # 不允许遗漏参数
    self.model.load_state_dict(sd, strict=True)

    print(
        f"原始权重: {original_count}个张量；"
        f"适配后加载: {len(sd)}个；strict=True 通过"
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，停止测试")

    if not CKPT.is_file():
        raise FileNotFoundError(CKPT)

    # 读取作者测试集划分
    with SPLIT.open(encoding="utf-8") as f:
        test = json.load(f)["test"]

    if len(test) != 1623:
        raise RuntimeError(f"测试集数量异常: {len(test)}")

    if len({row[0] for row in test}) != 1623:
        raise RuntimeError("测试集中存在重复样本ID")

    # 读取一次预处理文本，避免重复加载
    text_path = (INFERENCE_DIR / test[0][4]).resolve()

    text_data = torch.load(
        text_path,
        map_location="cpu",
        weights_only=True
    )

    missing_text = [
        row[0] for row in test if row[0] not in text_data
    ]

    missing_audio = [
        row[0] for row in test
        if not (INFERENCE_DIR / row[3]).is_file()
    ]

    if missing_text or missing_audio:
        raise RuntimeError(
            f"缺失文本: {len(missing_text)}, "
            f"缺失音频: {len(missing_audio)}"
        )

    if any(row[-1] not in LABELS for row in test):
        raise RuntimeError("存在非法情绪标签")

    print("数据检查通过: 1623条音频、1623条文本")

    # 输出CSV：支持断点续跑
    columns = [
        "index", "instance", "ground_truth",
        "prediction", "correct"
    ] + [f"prob_{x}" for x in LABELS]

    OUT.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    new_file = not OUT.exists() or OUT.stat().st_size == 0

    if not new_file:
        with OUT.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            if reader.fieldnames != columns:
                raise RuntimeError("已有CSV格式不匹配")

            existing = list(reader)

        if len(existing) > len(test):
            raise RuntimeError("已有结果超过1623条")

        for idx, row in enumerate(existing):
            if (
                int(row["index"]) != idx
                or row["instance"] != test[idx][0]
                or row["ground_truth"] != test[idx][-1]
                or row["prediction"] not in LABELS
            ):
                raise RuntimeError(
                    f"第{idx}条历史结果不匹配，禁止续跑"
                )

    print(f"已经完成: {len(existing)}条")
    print(f"剩余待测: {1623 - len(existing)}条")

    # 使用已验证成功的完整checkpoint加载方案
    EmotionInference._load_weights = load_complete_weights

    model = EmotionInference(
        model_dir=str(CKPT.parent),
        fold_idx=1,
        device="cuda",
        dataset="iemocap6",
        modal="multimodal"
    )

    with OUT.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)

        if new_file:
            writer.writeheader()
            f.flush()

        for i in tqdm(
            range(len(existing), len(test)),
            desc="IEMOCAP-6",
            unit="sample"
        ):
            sample = test[i]
            instance = sample[0]
            label = sample[-1]

            data = text_data[instance]
            audio = (INFERENCE_DIR / sample[3]).resolve()

            try:
                pred, probs = model.predict(
                    audio_path=str(audio),
                    text=(
                        data["input_ids"],
                        data["attention_mask"]
                    ),
                    return_probabilities=True
                )
            except Exception as e:
                raise RuntimeError(
                    f"样本{i + 1}/1623 ({instance})推理失败，"
                    "之前成功的结果已经保存"
                ) from e

            if (
                pred not in LABELS
                or abs(sum(probs.values()) - 1.0) > 1e-3
            ):
                raise RuntimeError(
                    f"模型输出异常: {instance}, {pred}, {probs}"
                )

            record = {
                "index": i,
                "instance": instance,
                "ground_truth": label,
                "prediction": pred,
                "correct": int(pred == label)
            }

            record.update({
                f"prob_{k}": probs[k] for k in LABELS
            })

            writer.writerow(record)
            f.flush()

    # 必须全部完成才计算指标
    with OUT.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 1623:
        raise RuntimeError(
            f"仅完成{len(rows)}/1623条，禁止计算最终指标"
        )

    y_true = [r["ground_truth"] for r in rows]
    y_pred = [r["prediction"] for r in rows]

    accuracy = accuracy_score(y_true, y_pred) * 100

    weighted = f1_score(
        y_true, y_pred, labels=LABELS, average="weighted"
    ) * 100

    macro = f1_score(
        y_true, y_pred, labels=LABELS, average="macro"
    ) * 100

    print("\n========== 全量测试结果 ==========")
    print(f"有效样本: 1623/1623")
    print(f"Accuracy:    {accuracy:.2f}%")
    print(f"Weighted F1: {weighted:.2f}%")
    print(f"Macro F1:    {macro:.2f}%")

    print("\n分类报告:")
    print(classification_report(
        y_true, y_pred, labels=LABELS, zero_division=0
    ))

    summary = {
        "samples": 1623,
        "accuracy": accuracy,
        "weighted_f1": weighted,
        "macro_f1": macro,
        "checkpoint": str(CKPT)
    }

    (OUT.parent / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print("\n逐条结果:", OUT)
    print("指标汇总:", OUT.parent / "summary.json")


if __name__ == "__main__":
    main()
