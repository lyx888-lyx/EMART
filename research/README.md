# EMART diagnostics (research branch)

Diagnosis first; no claimed improvement. Original loss behavior is the default.

## Phase 1: existing predictions (no CUDA)

From repo root:

~~~bash
python -m unittest discover -s research -p 'test_*.py' -v

python research/diagnose_predictions.py --dataset meld7 \
 --predictions test_results/mm_test_meld/meld7_fold1_test_results.csv \
 --output research_outputs/meld_existing

python research/diagnose_predictions.py --dataset iemocap6 \
 --predictions test_results/mm_test_iemocap/iemocap6_fold1_test_results.csv \
 --output research_outputs/iemocap_existing
~~~

Each run writes summary.json, per_class.csv, confusion_counts.csv,
confusion_row_normalized.csv, error_flows.csv, false_neutral.csv.
All computed metrics and ratios are on a 0-1 scale (e.g. 0.667 = 66.7%).
Historical mm_test prediction CSVs lack probability fields; this step cannot
compute calibration or uncertainty from those files. Treat existing test
results as descriptive exploration, not a basis for tuning new hyperparameters.

## Phase 2: matched Neutral-pair ablation (GPU)

In the original supcon implementation, --weak_pos controls different
terms inconsistently. Merely setting --weak_pos 0 is NOT a clean removal
of Neutral weak-positive pairs. This branch adds --neutral_pair_policy:
- original: original implementation (default).
- ignore: remove neutral/non-neutral pairs entirely from the emotion
  contrastive loss, neither positives nor negatives. Neutral/neutral
  same-class pairs stay strong positives.

Same seed/config, one change:

~~~bash
DATASET=meld7 POLICY=original SEED=8 GPU=0 bash bash/run_research_neutral.sh
DATASET=meld7 POLICY=ignore   SEED=8 GPU=0 bash bash/run_research_neutral.sh
~~~

For IEMOCAP:

~~~bash
DATASET=iemocap6 POLICY=original SEED=8 GPU=0 bash bash/run_research_neutral.sh
DATASET=iemocap6 POLICY=ignore   SEED=8 GPU=0 bash bash/run_research_neutral.sh
~~~

Runner defaults: MELD wav2vec2_0 10 epochs, IEMOCAP wavlm 20 epochs,
learning rate 2e-5, minimum LR 1e-6, weak_pos 0.3. Both conditions use
the same configuration. The new original run is the valid control,
not a historical checkpoint with different settings. Repeat paired seeds
8, 9, 10 for stability. Best model chosen using validation weighted F1.

Exports from best-validation epoch (including probabilities) are written to:

~~~text
finetune/meld7/multimodal/lr2e-05_ep10/research_predictions/
 research_meld7_wav2vec2_0_original_seed8/dev_fold_1.csv
 research_meld7_wav2vec2_0_original_seed8/test_fold_1.csv
 research_meld7_wav2vec2_0_ignore_seed8/dev_fold_1.csv
 research_meld7_wav2vec2_0_ignore_seed8/test_fold_1.csv
~~~

## Phase 3: paired comparison (compare DEV first)

~~~bash
PRED_ROOT=finetune/meld7/multimodal/lr2e-05_ep10/research_predictions
python research/diagnose_predictions.py --dataset meld7 \
 --predictions "$PRED_ROOT/research_meld7_wav2vec2_0_original_seed8/dev_fold_1.csv" \
 --compare "$PRED_ROOT/research_meld7_wav2vec2_0_ignore_seed8/dev_fold_1.csv" \
 --output research_outputs/meld_dev_original_vs_ignore_seed8
~~~

paired_summary.json lists original-wrong/ignore-right and vice versa plus
per-class correct counts. paired_predictions.csv contains matched sample
labels/predictions. The script verifies instance IDs and labels. After
analysis decisions are fixed, one-time comparison of test_fold_1.csv is
appropriate. Do not choose configurations using test-set results.

## Return results

Share output directories (summary.json, per_class.csv, error_flows.csv,
false_neutral.csv, paired_summary.json) and logs. Do not upload raw audio
or model checkpoints for the initial analysis.

This is an intervention on one mechanism, not proof of causality from one
seed. It does not test instance-level VAD geometry. Original training logs
test performance every epoch, but selects checkpoints by validation metrics:
ignore test metrics until all choices are finalized.
