import argparse
import json
import torch
from pathlib import Path

from inference import EmotionInference


# 替换权重加载方式，不改变模型结构或 forward
def load_full_checkpoint(self):
    ckpt_path = self.model_dir / f"iemocap_fold_{self.fold_idx}.pt"

    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    print(f"\nLoading COMPLETE checkpoint: {ckpt_path}")

    state = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=True
    )

    # 让 RoBERTa 的词嵌入大小与作者权重一致
    key = "text_model.semantic_model.embeddings.word_embeddings.weight"

    if key not in state:
        raise KeyError(f"Checkpoint 中缺少: {key}")

    saved_vocab = state[key].shape[0]
    current_vocab = (
        self.model.text_model.semantic_model
        .embeddings.word_embeddings.weight.shape[0]
    )

    if saved_vocab != current_vocab:
        print(f"Resize vocab: {current_vocab} -> {saved_vocab}")
        self.model.text_model.semantic_model.resize_token_embeddings(
            saved_vocab
        )

    # 严格加载：缺少参数、额外参数或形状不匹配都会报错

    # =====================================================
    # Compatibility: PyTorch weight_norm key naming
    # newer parametrizations.weight -> legacy weight_g/v
    # =====================================================
    prefix = "audio_model.backbone_model.encoder.pos_conv_embed.conv"

    mapping = {
        f"{prefix}.parametrizations.weight.original0":
            f"{prefix}.weight_g",
        f"{prefix}.parametrizations.weight.original1":
            f"{prefix}.weight_v",
    }

    expected = self.model.state_dict()

    for source_key, target_key in mapping.items():
        if source_key not in state:
            raise KeyError(f"Checkpoint missing: {source_key}")

        if target_key not in expected:
            raise KeyError(f"Model missing: {target_key}")

        if state[source_key].shape != expected[target_key].shape:
            raise RuntimeError(
                f"Shape mismatch: {source_key} "
                f"{tuple(state[source_key].shape)} vs "
                f"{target_key} {tuple(expected[target_key].shape)}"
            )

        if target_key in state:
            raise RuntimeError(f"Key collision: {target_key}")

        state[target_key] = state.pop(source_key)
        print(f"Compatible key mapping: {source_key} -> {target_key}")

    # =====================================================
    # Compatibility: RoBERTa position_ids buffer
    # Verify standard position indices before dropping
    # =====================================================
    pos_key = "text_model.semantic_model.embeddings.position_ids"

    if pos_key in state and pos_key not in expected:
        position_ids = state[pos_key].cpu()

        standard_ids = torch.arange(
            position_ids.shape[-1],
            dtype=position_ids.dtype
        ).unsqueeze(0)

        if (
            position_ids.shape != standard_ids.shape
            or not torch.equal(position_ids, standard_ids)
        ):
            raise RuntimeError(
                "Non-standard position_ids; refusing to discard"
            )

        del state[pos_key]
        print("Removed verified non-trainable position_ids buffer")

    # =====================================================
    # Strict loading: no trainable parameters may be missed
    # =====================================================
    self.model.load_state_dict(state, strict=True)


    print(f"完整加载成功：{len(state)} 个参数/缓冲区张量")
    print("strict=True 检查通过")


EmotionInference._load_weights = load_full_checkpoint


parser = argparse.ArgumentParser()
parser.add_argument("--full", action="store_true")
args = parser.parse_args()

model_dir = "../finetune/iemocap6/multimodal/pretrained"

if not args.full:
    # 仅测试一条样本
    model = EmotionInference(
        model_dir=model_dir,
        fold_idx=1,
        device="cuda",
        dataset="iemocap6",
        modal="multimodal"
    )

    with open(
        "../train_split/train_split_iemocap/iemocap6_fold1.json"
    ) as f:
        data = json.load(f)

    sample = data["test"][0]
    pt = torch.load(
        sample[4],
        map_location="cpu",
        weights_only=True
    )

    item = pt[sample[0]]

    pred, probs = model.predict(
        audio_path=sample[3],
        text=(item["input_ids"], item["attention_mask"]),
        return_probabilities=True
    )

    print("\n===== 单条推理结果 =====")
    print("样本ID:", sample[0])
    print("真实标签:", sample[-1])
    print("预测标签:", pred)
    print("各类别概率:", probs)

else:
    # 使用官方测试函数运行完整测试集
    test_args = argparse.Namespace(
        model_dir=model_dir,
        fold_idx=1,
        device="cuda",
        dataset="iemocap6",
        modal="multimodal",
        split_data_dir="train_split/train_split_iemocap",
        text_csv_path=(
            "../train_split_gen/previous_utt_csv/"
            "previous_current_dynamic_self_other_256/IEMOCAP_test.csv"
        ),
        output_dir="../test_results/iemocap6_full_ckpt"
    )

    from test_inference import run_test_inference
    run_test_inference(test_args)
