> Upstream competition guide copied for provenance. Paths beginning `go/` refer to the backing competition repository. Start at this course project’s `docs/REPRODUCTION.md` for its working commands.

# Organizer reproduction package

Accepted Task1 output: **0.8721950**, `g23_greplace.csv`, byte-identical to the restored last upload `best_872195.csv`.

This directory provides a clean-workspace **saved-model replay**, its pinned dependencies and an official-data path checker. The compact replay bundle includes the fitted policy MLP and LDA, declared extracted policy features, coordinate-fusion inputs, original-family parent predictions, exact runtime source, model configuration and notices. It deliberately contains no official CSVs or training dataset. All dependencies are identified by role and SHA-256 in its manifest.

**This is not yet a verified raw-CSV-only pipeline.** The inherited legacy base-view reconstruction discrepancy remains open. The supplied intermediates are explicitly declared; their presence must not be described as proof that every parent model was regenerated from the organizer's raw data. See the acceptance checklist below.

## A. Install and reproduce the exact saved-model result

Use Linux/Python3.11.17. The tested server environment executes PyTorch2.6.0+cu124 on CPU; no GPU is needed for this command. Approximate uncompressed payload is35MB, plus the Python environment and outputs. Private GitHub access is needed only for download; an organizer can instead receive the same archive directly with its manifest and checksum.

```bash
# Start from a checkout of this repository, branch codex/task1-reproduction-20261117.
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r go/organizer/requirements-replay.txt
mkdir -p organizer_assets
gh release download go-organizer-package-20261124 \
  --repo goog-msft-fb-nflx-nvda-aapl/aicup2026-solar-defect \
  --pattern organizer_saved_model_replay1124.tar.gz --dir organizer_assets
python go/organizer/reproduce.py \
  --archive organizer_assets/organizer_saved_model_replay1124.tar.gz \
  --workspace /absolute/path/to/EMPTY-organizer-workspace
```

The workspace must be absent or empty. The runner rejects unsafe archive members, checks every saved input hash and recomputes the MLP embeddings, prototype scores and expanded Gaussian fusion. The actual accepted output is not copied from a bundled accepted CSV: the best CSV is generated, and then checked against its expected SHA-256. Original-family rows are inherited from the explicitly archived parent CSV.

Output:

```
EMPTY-organizer-workspace/reproduction/best1123/g23_greplace.csv
EMPTY-organizer-workspace/organizer_replay_receipt.json
```

Expected prediction SHA-256:

```
eac2f9120aff4dd74102d08b2cbcc3e2cf9009d976fd80a63890b371deacde26
```

Archive SHA-256:

```
b0b6f144a3c0961566472f680f72112e59df236f273a69aba60af57007f404c8
```

A fresh-directory extraction/replay is recorded in `go/reproduction/20261124/organizer_audit/`. This tests restored filesystem isolation using the recorded server Python environment; it does not claim a newly installed venv or raw-data-only verification. Reusing an extracted workspace is supported by omitting `--archive`; all inputs are checked again before running.

## B. Organizer-provided dataset and raw extraction commands

`../ORGANIZER_REPRODUCTION.md` gives the directory tree. With original CSVs placed there:

```bash
export PROJECT_ROOT=/absolute/path/to/raw-reproduction-workspace
python go/organizer/check_official_data.py --root "$PROJECT_ROOT" \
  --inventory go/reproduction/20261123_cleanup/official_data_inventory.json
export GO_ROOT="$PROJECT_ROOT"
cd "$PROJECT_ROOT/src"
SUBSAMPLE=1 python prep.py
```

Retain the organizer's original filenames and ordering. `data/train_arr.npz` is regenerated locally; neither the raw CSVs nor a second copy of that cache is backed up.

For policy features, install broader training requirements (`go/requirements.txt` and exact environment freezes), restore the licensed HumanSL network at `PROJECT_ROOT/katago/humanv0.ckpt` and pinned KataGo source at `PROJECT_ROOT/katago/kgsrc` from `go-policy-residual-pilot-20261119`. Source commit is `d91ea855110dae533f0aada947b2b7d78cc8a4e1`. Run these commands in a fresh training/extraction workspace without the corresponding output directories:

```bash
CUDA_VISIBLE_DEVICES=0 python "$PROJECT_ROOT/src/codex_policy_test_features_1119.py" --root "$PROJECT_ROOT" --pool pub2
CUDA_VISIBLE_DEVICES=0 python "$PROJECT_ROOT/src/codex_policy_test_features_1119.py" --root "$PROJECT_ROOT" --pool priv2
CUDA_VISIBLE_DEVICES=0 python "$PROJECT_ROOT/src/codex_policy_residual_pilot_1119.py" --root "$PROJECT_ROOT" --training --players 1000
CUDA_VISIBLE_DEVICES=0 python "$PROJECT_ROOT/src/codex_policy_residual_pilot_1119.py" --root "$PROJECT_ROOT" --players 600
CUDA_VISIBLE_DEVICES=0 python "$PROJECT_ROOT/src/codex_policy_expand1122.py" --root "$PROJECT_ROOT" --training --players 2500 --shard 0
CUDA_VISIBLE_DEVICES=0 python "$PROJECT_ROOT/src/codex_policy_expand1122.py" --root "$PROJECT_ROOT" --training --players 2500 --shard 1
GO_ROOT="$PROJECT_ROOT" CUDA_VISIBLE_DEVICES=0 python "$PROJECT_ROOT/src/codex_policy_metric1119.py"
python "$PROJECT_ROOT/src/codex_policy_expand_fit1122.py" --root "$PROJECT_ROOT"
```

The fitting scripts additionally need aligned `results/HO_meta.npz`, heldout coordinate arrays `HO_{a2,b1,b2,b3,h4,h5}.npy` and the documented parent pipeline. The script source and frozen validation configurations are authoritative; stochastic retraining is not expected to produce byte-identical weight files.

Coordinate-model preprocessing, pretraining, all member commands, rank-gate and fusion settings are documented in `../TRAINING_CONFIGS.md`, `../REPRODUCE_TASK1_1117.md`, `../REPRODUCE_TASK1_1118.md`, `../REPRODUCE_TASK1_CX35.md`, `../REPRODUCE_TASK1_CX38.md` and `../REPRODUCE_TASK1_BEST_20261123.md`. Exact archived checkpoint/config inventories are indexed in `asset_index.json`; they must remain available alongside this repository. These commands are a documented reconstruction route, **not** a passing end-to-end raw-data acceptance test.

## C. New experiments and historical source

The four fixed new candidates, local validation, selected covariance model, source and replay evidence are in `go-final-four-20261124`. The three tested Nov24 candidates and their actual negative results are under `go/reproduction/20261124/submission_requests`. Final restoration is under `go/reproduction/20261124/best_restoration`. No automatic upload is part of organizer reproduction.

All current server scripts plus production/non-production configurations, command records, logs and submissions are frozen in `go-cleanup-audit-20261123`, with3,529 file hashes. Historical trained model archives remain in the indexed private releases. No passwords or website cookies are required for inference or included in the delivery package.

## D. Acceptance and report status

| Requirement | Evidence/status |
| --- | --- |
| Official data placement and preprocessing command | Documented; CSV hash checker provided |
| Accepted source, model configuration, fitted policy weights and expected CSV | Archived and hash-indexed |
| Clean restored saved-model replay | Byte-exact in empty workspace; receipt archived |
| Full raw-CSV reconstruction of accepted parent | **Open: legacy base-view mismatch** |
| Retraining source, settings and parent model weights | Archived, dependency-indexed; stochastic training caveat |
| New/failed experiment source and real feedback | Archived |
| External HumanSL use, source pin and license | Declared and preserved |
| Report method/score/source accuracy | Updated draft; old misleading exclusion text archived as history |
| Complete official roster, adviser and contributions | Team must confirm before delivery |
| Organizer access and actual report/source submission | Must be checked separately; private GitHub backup is not delivery |

The final report cannot honestly assert complete raw-data reproduction while the open item remains. The required report must be finalized and submitted before the official cutoff; no permission to omit fields or submit late is inferred from this package.
