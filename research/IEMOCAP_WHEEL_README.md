# IEMOCAP: auditing EMART's fixed emotion-wheel prior (phase 1)

**Research question**: How well does EMART's fixed class-level angular ordering agree with independently annotated utterance-level Valence/Activation (VA) geometry, and where do their relationships disagree? Is disagreement *associated* with existing EMART errors?

**This branch does not change the EMART model, checkpoints, classifier, or contrastive loss.** It supplies observation-based diagnostics. Do not conclude that the wheel loss causes errors from these correlations alone. The code is pure Python standard library (no GPU or pip dependencies required).

## 0. Requirements and data inspection

Run from the root of the **EMART** repository. Existing train/dev/test split:

```text
train_split/train_split_iemocap/iemocap6_fold1.json
```

By default, the program expects the *original IEMOCAP human emotion annotations* under a sibling directory:

```text
../dataset/IEMOCAP_full_release/
  Session1/dialog/EmoEvaluation/*.txt
  Session2/dialog/EmoEvaluation/*.txt
  ...
  Session5/dialog/EmoEvaluation/*.txt
```

These text files contain an utterance ID, discrete emotion label, and the original human-rated `[valence, activation, dominance]` triple. This is **not** the tokenized `.pt` file or an audio feature and is **not** supplied by a standard EMART inference CSV. If they are missing, **do not synthesize VAD using the EMART predictions**; obtain the IEMOCAP annotations or supply separately curated scores.

Locate them before running:

```bash
# Both dataset and IEMOCAP_full_release may be symlinks.
find -L ../dataset/IEMOCAP_full_release -type d -name EmoEvaluation

# Check where the symlink resolves:
readlink -f ../dataset/IEMOCAP_full_release
```

If the directory is elsewhere, supply `--iemocap-root /actual/path/IEMOCAP_full_release`.

If you already have an utterance-keyed CSV, you may supply `--vad-csv /path/to/your_vad.csv` instead. Required fields (case-insensitive): `instance,valence,activation,dominance`. Alternative names include `utterance_id,val,arousal,dom`. **The instance ID must be exactly the IEMOCAP utterance ID** (e.g. `Ses05M_script03_2_F001`). Ratings should be actual human annotations on the dataset's original 1–5 scale. Do not copy a category's fixed prototype into each sample: that would make our test circular.

## 1. Unit tests (no GPU)

```bash
python -m unittest discover -s research -p 'test_*.py' -v
```

The synthetic tests cover annotation parsing, matching to the JSON split, circular angle arithmetic, prototype ranking, class overlap and a full mini end-to-end audit.

## 2. Create the joined human VAD manifest

```bash
python research/iemocap_wheel_validity.py prepare \
  --iemocap-root ../dataset/IEMOCAP_full_release \
  --split-json train_split/train_split_iemocap/iemocap6_fold1.json \
  --out research_outputs/iemocap_wheel/manifest.csv
```

Or replace `--iemocap-root ...` with `--vad-csv /path/to/vad.csv`.

Outputs:

- `research_outputs/iemocap_wheel/manifest.csv`: matched ID, split, label, Valence, Activation, Dominance.
- `coverage.json`: how many of **train/dev/test** matched; **must inspect coverage and label mismatch count**.
- `missing_vad.csv`: utterances absent from original ratings.
- `label_mismatches.csv`: discrete labels incompatible with the current split, excluded from the analysis rather than silently used.

The split JSON may encode the final emotion field as a category code like neu/sad/fru/ang/hap/exc rather than an integer; the parser supports both formats.\n\nThe command exits nonzero when any split has below 80% coverage (changeable with `--min-coverage`, but lowering this threshold requires a research justification). A nonzero exit may still leave diagnostic files for you to inspect.

## 3. Analyze DEV first: independent VA geometry without prediction labels

```bash
python research/iemocap_wheel_validity.py analyze \
  --manifest research_outputs/iemocap_wheel/manifest.csv \
  --split dev \
  --dims VA \
  --output research_outputs/iemocap_wheel/dev_geometry
```

This uses **training-split** human VA centroids to evaluate whether the wheel's five non-neutral label angles are consistent with **dev-split instance-level VA** relationships. It does not need a model checkpoint.

Primary outputs:

- `train_class_centroids.csv`: human-annotated VA(D) class centers and SDs estimated on train only.
- `class_pair_relationship.csv`: each of the ten non-neutral class pairs, paper's circular angular distance, and the human train VA centroid distance.
- `triplet_per_anchor.csv`: for dev anchors, fraction of sampled cross-class triplets where the **paper angle ordering** disagrees with human VA distance ordering.
- `same_vs_cross_class.csv`: for each ordered category pair, frequency with which an utterance from a **different class** is closer in VA than a second utterance from the **same class**.
- `per_instance.csv`: dev utterance VA, and proportion of cross-class prototype-ranking inversions relative to paper angle ordering.
- `summary.json`: sampled statistics and explicit warnings.

**Neutral** has no assigned angle in EMART, so it is retained in descriptive class statistics but excluded from angular-distance and triplet ranking tests. The primary 2D comparison is **VA** (Valence + Activation), not Dominance; `--dims VAD` is optional sensitivity analysis, and its 3D distances should not be treated as angles on a 2D circle.

## 4. Optional retrospective TEST error association (exploratory, not model selection)

The repository has an existing IEMOCAP prediction CSV (best-run Accuracy around 70.79%) with IDs and actual predictions:

```bash
python research/iemocap_wheel_validity.py analyze \
  --manifest research_outputs/iemocap_wheel/manifest.csv \
  --split test \
  --dims VA \
  --predictions test_results/mm_test_iemocap/iemocap6_fold1_test_results.csv \
  --output research_outputs/iemocap_wheel/test_existing_prediction_audit
```

The program checks the prediction truth label against the VAD-joined split's truth label by utterance ID and **stops on a mismatch**, rather than silently joining different folds. Records where human VAD is unavailable are not fabricated.

Relevant additional outputs:
- `error_by_class_and_mismatch.csv`: per-label accuracy error rate grouped by *instance-level angle-vs-human-VA ranking disagreement*, plus within-class VA distance.
- `high_mismatch_errors.csv`: small list of misclassified, high-disagreement utterance IDs for targeted follow-up. No raw audio or text is exported.

The predicted error correlations are *observational* and may be confounded by class, label disagreement or atypical VA expressions. Do not claim that EMART's wheel caused an error without a matched model trained **without L_emo**, trained on the same split/seed. In particular, **a high disagreement rate alone is not a new contribution**: what matters is whether it predicts a differential effect of the wheel loss.

## 5. What to return

After phases 1–3, share:

```bash
zip -r iemocap_wheel_phase1.zip research_outputs/iemocap_wheel
```

If you have the existing test predictions and wish to audit them, run phase 4 and rezip. Send the zip file and the terminal test output. **No need to upload source audio, transcripts, or checkpoints.**

## Reading the evidence responsibly

1. Differences between fixed angles and VA scores demonstrate *lossy prior representation*, not that the loss is harmful.
2. Human ratings have noise and are ordinal/limited in resolution; Euclidean VA distance is an **operational proxy**, not a unique psychological ground truth.
3. Class-pair comparisons have only ten relationships and shared categories; sampled triplets reuse utterances. Treat their correlations as descriptive rather than independent-sample significance tests.
4. Train-derived centroids avoid fitting a new geometry on held-out data. If later changing any model or angle hyperparameter, select on **dev**, not test.
5. In an actual causal follow-up, compare `L_cls + L_al` versus `L_cls + L_al + L_emo` with matched training protocol and multiple random seeds, and test whether high-prior-conflict utterances respond differently.

The code lives at `research/iemocap_wheel_validity.py`, with tests at `research/test_iemocap_wheel_validity.py`. This branch inherits the earlier `research/emart-diagnostics-v1` scripts but leaves them unchanged.
