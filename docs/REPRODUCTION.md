# Reproduction

## 1 Official inputs

Obtain the authorized AI CUP 2026 Task 1 v1.1 training data and updated November11 test files from the [official competition](https://www.aidea-web.tw/topic/dc0b799e-690f-4e84-a597-93e68d8ba688). The official data is not bundled here. Archive SHA-256: `e0880a3e8723f776e931ecce7fbc09eace9a0dd42b18a1d8ebac4a1628d2395e`.

Use an absolute `PROJECT_ROOT` directory with this layout:

```text
PROJECT_ROOT/
  src/                         model_src/*.py copied here
  data/training/               official training CSVs
  data/tests1v2/tests/          updated official test/candidate CSVs
  results/                     restored frozen parent inputs
  reproduction/                restored release paths below
  katago/humanv0.ckpt           licensed pretrained network, for raw extraction
  katago/kgsrc/python/          pinned KataGo source, for raw extraction
```

Model inference was tested on the existing Python3.11 / PyTorch2.6 environment on a shared H200 server. The recorded package versions are in `model_environment.json`. Document-building dependencies are separate. Use an isolated environment compatible with the recorded CUDA version. Loading the frozen checkpoints requires trusted inputs; do not substitute untrusted weights.

## 2 Restore the frozen assets

The backing repository is private: `goog-msft-fb-nflx-nvda-aapl/aicup2026-solar-defect`. An authorized grader must have access to the source and release assets. Existing credentials are never bundled in this course project.

| Release | Required material |
|---|---|
| `task1-reproduction-20261117` | initial parent model, feature, environment and dependency backups |
| `go-best-20261118` | frozen gate models and auxiliary score dependencies |
| `go-cx35-experiment-20261118` | selected cx35 model and score delta |
| `go-review-pipeline-20261118` | selected cx38 checkpoint, training and inference artifacts |
| `go-best-noverifier-20261119` | accepted .8630397 parent and its reproduction guide |
| `go-policy-residual-pilot-20261119` | licensed HumanSL model/source, pilot features and notice |
| `go-policy-candidate-20261119` | full new-family test features and metadata |
| `go-policy-metric-20261120` | selected MLP checkpoint, scaler and training evidence |
| `go-policy-setmatch-20261121` | selected LDA checkpoint and Gaussian matching code |
| `go-next-experiments-20261119` | cx40 training and inference artifacts |
| `go-best-20261122` | accepted prediction, receipt, verification and source snapshot |

Keep archive-relative paths intact. Follow the full frozen `REPRODUCE_TASK1_BEST_20261122.md` and linked parent guides supplied with the backing source release. The course copy is `docs/FROZEN_BEST_GUIDE.md`.

Required inputs for the final saved-model command include:

```text
reproduction/cx38_candidate_pipeline1118/sub/go19_nover.csv
reproduction/cx38_candidate_pipeline1118/results/S_{pub2,priv2}_pids.npy
reproduction/policy_test1119/{pub2,priv2}_features.npz
reproduction/policy_metric1119/policy_metric.pt
reproduction/policy_setmatch1121/policy_lda.npz
reproduction/experiments_1119/runs/cx40_lm4_noaug_all/results/
    Sfinal_newmember_candidate_{pub2,priv2}.npy
```

Braces indicate two separate filenames, not a literal path. Restore frozen parent results as well because the control check requires `Sfinal_review_candidate_{pub2,priv2}.npy`.

## 3 Saved-model inference

```bash
export PROJECT_ROOT=/absolute/path/to/restored/project
mkdir -p "$PROJECT_ROOT/src"
cp model_src/*.py "$PROJECT_ROOT/src/"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  python "$PROJECT_ROOT/src/codex_reproduce_best1122.py" \
  --root "$PROJECT_ROOT" --device cuda
```

Choose a GPU with adequate capacity. The script regenerates MLP embeddings and Gaussian scores from saved models and policy features, checks the previous control, fuses them with the frozen cx40 coordinate scores, and writes `reproduction/best1122/g22_cx40.csv`.

Expected prediction SHA-256:

```text
123a07fc353790bc983a76862d10db96e03399df4160a3e7af44deb293b61fdb
```

The script rejects a mismatch and writes `inference_verification.json`. This exact inference was checked on the server before the frozen best release was created. A copied evidence record is in `results/best_inference_verification.json`.

## 4 Raw extraction and training

Coordinate training source, language-model initialization, selected checkpoint and commands are in the cx40 training archive. Frozen HumanSL extraction uses `codex_policy_residual_pilot_1119.py` for training/holdout and `codex_policy_test_features_1119.py` for test pools. `codex_policy_metric1119.py` trains the MLP; `codex_policy_lda1120.py` and `codex_policy_setmatch1121.py` fit/evaluate the LDA and reference matching. Run the frozen preprocessing first to obtain `data/train_arr.npz`, and restore the six heldout embedding caches and `HO_meta.npz` for local comparisons. These are dependencies of selection, not test labels.

The expanded study uses two `codex_policy_expand1122.py --training --shard {0,1}` processes followed by `codex_policy_expand_fit1122.py`. It adds 5,000 disjoint training players to the existing 1,000, selecting 15 games per player. Its snapshot had no public evaluation yet.

## 5 Scope and known limitation

Saved-model inference using frozen coordinate scores and extracted policy features reproduces the scored CSV exactly. The legacy base-view component still changes some lower ranks when reconstructed from raw inputs. The separate raw reconstruction scored .8670060; it is not the accepted best. Consequently full raw-input equality is unresolved. The course materials disclose this limitation rather than treating an archived prediction alone as full training reproducibility.
