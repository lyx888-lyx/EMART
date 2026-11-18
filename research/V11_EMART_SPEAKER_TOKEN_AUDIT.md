# V11: EMART IEMOCAP implementation fidelity before new architecture

Date: 2026-11-18. Status: code committed; no local data/GPU run by assistant.

Paper: Emotion-Wheel-Guided Audio-Referred Text Representation for
Multimodal Emotion Recognition in Conversation, ACL 2026 Long,
https://aclanthology.org/2026.acl-long.1875/. Table 4 reports best
of three seeds, IEMOCAP WavLM 70.93 W-F1 / 70.79 Acc and Wav2vec
68.61 / 68.39; MELD WavLM 64.19 / 66.36 and Wav2vec 64.71 / 66.70.
These are paper TEST results, not interchangeable with our CSS
session-4 development results or with its different modalities.

## Two concrete source observations (local cache still must be audited)

1. train_split_gen/iemocap_preprocessing.py treats row['Speaker']
   as utterance ID for the cache, but compares entire ID to determine
   historical SELF/OTHER. If the IDs follow Ses01F_impro01_F000 format,
   same-person different turns are incorrectly labeled OTHER.
   The speaker is the final F/M preceding the utterance number,
   not the recording prefix gender.
2. That script registers [Current], SELF, OTHER; model/custom_roberta.py
   registers [Previous], [Current], SELF, OTHER. Depending on actual
   tokenizers/cache generation, IDs need not agree. Model curr_only
   pooling uses the model's [Current] numeric ID, so mismatch can
   affect current span selection as well as role semantics.

A previous local fix may already have resolved one or both. Do not
claim a bug in the user's cached data until exact token audit agrees.
The audit reconstructs the original chronological context-selection
logic from TRAIN/DEV CSVs using original tokenization; the aligned
arm changes numeric IDs only, role_fixed changes actual historical
role tokens only using the same selected utterances/order/context length.
For each original selected utterance validate exact token/mask matches,
source label agreement, existence of audio, full-utterance identities,
consistent dialogue grouping and train/dev dialogue disjointness.
The same model-current numeric ID may correspond to legacy SELF:
presence of that numeric ID alone is NOT evidence of semantic alignment.

If local cache matches legacy exactly: compare legacy/aligned/role_fixed.
If aligned exactly: compare aligned/role_fixed.
If already role_fixed, no relevant role change, or unmatched: STOP.
No arbitrary cache reconstruction as a supposedly reproduced baseline.
Prepared data references are local to the user machine and never pushed.
No official TEST input generation, loader or score. Original training
code selects checkpoints by DEV but scores TEST every epoch; the
new separate runner only constructs TRAIN/DEV loaders and uses
the original train_epoch and validate_epoch functions. Old code intact.

## Run on server

From your existing EMART checkout and ORIGINAL EMART environment:

    git fetch origin
    git switch experiment/v11-speaker-token-audit
    python -m unittest discover -s tests -p 'test_v11_roles.py' -v
    python -u train_split_gen/v11_audit_and_prepare.py

If needed supply --csv-root /path/to/telme_data and
--split-json /path/to/iemocap6_fold1.json. Defaults try repo-local
or sibling dataset directories; audit does not change data paths.
The original manifest's relative audio/cache paths are resolved
from experiment/, as in original finetune.py. No need to reinstall
the previously working EMART dependencies. Do not use CSS's cmdpad
environment unless that was also the previous EMART environment.

Output results/v11_roles/audit.json. If audit allows training:

    bash bash/run_v11_dev.sh --smoke
    bash bash/run_v11_dev.sh

All arms use WavLM, same original RoBERTa, audio waveforms, model,
optimizer, complete loss, scheduler, seed8, batch64, 20 epochs,
original fold1 train/dev. Preserve source hyperparameters (including
min_lr > initial lr; do not silently alter optimizer for one arm).
Epoch selected by DEV weighted F1; true macro F1 separately reported.
No test-based selection. Original dev split differs from CSS
session-4 split, and need not be held-out-session speaker-disjoint.
Smoke is one epoch and not performance evidence. Result directories
are separate; no overwrites. Code not yet GPU-tested.

## Interpretation

Tokenizer alignment and speaker role correction are implementation
fidelity changes, not paper novelty. A gain would establish a better
matched baseline, not prove cross-speaker emotion causation.
Only after the corrected baseline is established, design an actual
method: explicitly learn current speech relative to prior same-
speaker speech and intervening other-speaker speech, rather than
just role-marked text. Verify raw chronological timestamps, use
identical repeated-turn support and capacity-matched controls.
Existing literature must be inspected before any novelty claim.
