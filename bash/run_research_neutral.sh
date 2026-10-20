#!/usr/bin/env bash
# Usage: DATASET=meld7 POLICY=ignore SEED=8 GPU=0 bash bash/run_research_neutral.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${DATASET:-meld7}"
POLICY="${POLICY:-original}"
SEED="${SEED:-8}"
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONNOUSERSITE=1

case "$DATASET" in
  meld7)
    AUDIO_MODEL="${AUDIO_MODEL:-wav2vec2_0}"
    EPOCHS="${EPOCHS:-10}"
    BATCH_SIZE=32
    POOLING=curr_only
    MAX_TXT_LEN=128
    SPLIT_DIR=train_split/train_split_meld
    ;;
  iemocap6)
    AUDIO_MODEL="${AUDIO_MODEL:-wavlm}"
    EPOCHS="${EPOCHS:-20}"
    BATCH_SIZE=64
    POOLING=cls
    MAX_TXT_LEN=256
    SPLIT_DIR=train_split/train_split_iemocap
    ;;
  *)
    echo "DATASET must be meld7 or iemocap6" >&2
    exit 2
    ;;
esac
if [[ "$POLICY" != "original" && "$POLICY" != "ignore" ]]; then
  echo "POLICY must be original or ignore" >&2
  exit 2
fi

RUN_NAME="research_${DATASET}_${AUDIO_MODEL}_${POLICY}_seed${SEED}"
SETTING="lr2e-05_ep${EPOCHS}"
# finetune.py opens a logging file before initializing experiment directories.
mkdir -p "${ROOT}/experiment/finetune/${DATASET}/multimodal/${SETTING}/${RUN_NAME}"
echo "Dataset=$DATASET, Policy=$POLICY, Seed=$SEED, Audio=$AUDIO_MODEL"
echo "Run name=$RUN_NAME"
echo "Best checkpoint selected on validation weighted F1."
cd "${ROOT}/experiment"
python finetune.py \
  --dataset "$DATASET" \
  --text_model roberta-base \
  --audio_model "$AUDIO_MODEL" \
  --modal multimodal \
  --pooling_mode curr_only \
  --multimodal_pooling "$POOLING" \
  --speaker_mode self_other \
  --split_data_dir "$SPLIT_DIR" \
  --load_pt \
  --finetune_roberta True \
  --batch_size "$BATCH_SIZE" \
  --max_txt_len "$MAX_TXT_LEN" \
  --num_epochs "$EPOCHS" \
  --learning_rate 2e-5 \
  --min_lr 1e-6 \
  --weight_decay 0.0001 \
  --best_metric mf1 \
  --ssl_mode supcon \
  --weak_pos 0.3 \
  --at_barlow_align True \
  --plutchik_instance_coeff 0.5 \
  --plutchik_instance_match True \
  --plutchik_ipc_grad_start_ep 5 \
  --neutral_pair_policy "$POLICY" \
  --research_export_predictions \
  --seed "$SEED" \
  --exp_dir "$RUN_NAME"
echo "Results:"
echo "${ROOT}/finetune/${DATASET}/multimodal/${SETTING}/research_predictions/${RUN_NAME}/"
