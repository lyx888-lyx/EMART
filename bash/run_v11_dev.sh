#!/usr/bin/env bash
# Run only the arms supported by the source/cache audit, same WavLM model.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
PY="${EMART_PYTHON:-python}"
PREP="${V11_PREPARED:-$ROOT/results/v11_roles}"
if [[ ! -f "$PREP/audit.json" ]]; then
  echo "Run train_split_gen/v11_audit_and_prepare.py first." >&2
  exit 1
fi
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--smoke" ) ]]; then
  echo "Usage: bash bash/run_v11_dev.sh [--smoke]" >&2; exit 1
fi
ARMS="$("$PY" - "$PREP/audit.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
print(" ".join(r["training_arms"]))
PY
)"
if [[ -z "$ARMS" ]]; then
  echo "Audit allows no new arms. Send audit.json; do not train." >&2
  exit 1
fi
EP=20
EXTRA=()
STAGE=full
if [[ "${1:-}" == "--smoke" ]]; then
  EP=1
  EXTRA=(--v11-smoke)
  ARMS=role_fixed
  STAGE=smoke
fi
for ARM in $ARMS; do
  if [[ -e "$ROOT/results/v11_emart/$STAGE/$ARM" ]]; then
    echo "Refusing to overwrite existing $STAGE/$ARM." >&2
    exit 1
  fi
done
mkdir -p "$ROOT/results/v11_emart/logs"
cd "$ROOT/experiment"
for ARM in $ARMS; do
  "$PY" -u v11_train_dev.py --v11-arm "$ARM" --v11-prepared "$PREP" "${EXTRA[@]}" \
    --dataset iemocap6 --audio_model wavlm --text_model roberta-base \
    --modal multimodal --pooling_mode curr_only --multimodal_pooling cls \
    --speaker_mode self_other --load_pt --finetune_roberta True \
    --batch_size 64 --max_txt_len 256 --num_epochs "$EP" \
    --learning_rate 2e-5 --min_lr 5e-5 --weight_decay 0.0001 \
    --best_metric mf1 --ssl_mode supcon --weak_pos 0.3 \
    --at_barlow_align True --plutchik_instance_coeff 0.5 \
    --plutchik_instance_match True --plutchik_ipc_grad_start_ep 5 \
    --seed 8 --print_verbose False --wandb False \
    2>&1 | tee "$ROOT/results/v11_emart/logs/${STAGE}_${ARM}.log"
done
"$PY" - "$ROOT/results/v11_emart/$STAGE" $ARMS <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
rows=[json.loads((root/m/"report.json").read_text()) for m in sys.argv[2:]]
assert all(r["utterance_ids"]==rows[0]["utterance_ids"] and r["labels"]==rows[0]["labels"] for r in rows)
b=rows[0]["dev"]["weighted_f1"]
print("\nV11 MATCHED DEV REPORT | no TEST")
for r in rows:
    d=r["dev"]
    print(f"{r['arm']:12s} W-F1={d['weighted_f1']:.2f}% delta={d['weighted_f1']-b:+.2f}pp "
          f"Macro-F1={d['macro_f1']:.2f}% ACC={d['accuracy']:.2f}% "
          f"UAR={d['uar']:.2f}% epoch={r['selected_epoch']}")
PY
