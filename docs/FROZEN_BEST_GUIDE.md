# Task1 best on November23 — 0.8721950

The fifth November23 upload, **g23_greplace.csv**, scored **0.8721950** and placed TEAM_10990 at **19/127** in the captured leaderboard. All five daily slots were used. This final exploratory upload is also the strongest known artifact. November24's final website upload is **manual by the user**.

Prediction SHA-256:

```text
eac2f9120aff4dd74102d08b2cbcc3e2cf9009d976fd80a63890b371deacde26
```

## Saved models and input placement

Use the recorded Python3.11.17 / PyTorch2.6.0+cu124 environment; CPU inference has also been verified in this same installation. The existing parent guides record full package versions. Set PROJECT_ROOT to an absolute restored workspace. Copy this release's `go/src/` into PROJECT_ROOT/src.

Restore the following private release assets, preserving archive-relative paths:

- `go-best-20261122` and its linked parent dependencies: original-family prediction template, parent controls and LM4/cx40 fusion. Full restoration is documented in REPRODUCE_TASK1_BEST_20261122.md.
- `go-next-experiments-20261119`: cx40 training and inference archives. Required coordinate arrays are `reproduction/experiments_1119/runs/cx40_lm4_noaug_all/results/Sfinal_newmember_candidate_{pub2,priv2}.npy`.
- `go-policy-candidate-20261119`: `reproduction/policy_test1119/{pub2,priv2}_features.npz`, including question IDs, player ordering and reference/query indexing.
- `go-policy-metric-20261120`: `reproduction/policy_metric1119/policy_metric.pt`, containing the selected MLP and its fitted scaler.
- `go-policy-expansion-20261122`: `reproduction/policy_expanded_lda1122/policy_lda.npz`, containing the 6,000-player LDA projection and its fitted scaler. The two training feature shards, training-source version and selection evidence are preserved in that release.
- `go-tomorrow-five-20261123`: candidate manifest, settings, input hashes, source and independent five-CSV replay evidence.

The parent control additionally requires `reproduction/cx38_candidate_pipeline1118/sub/go19_nover.csv`, `results/Sfinal_review_candidate_{pub2,priv2}.npy` and `results/S_{pub2,priv2}_pids.npy`. Braces indicate two filenames, not a literal path.

## Exact prediction command

```bash
export PROJECT_ROOT=/absolute/path/to/restored/project
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python "$PROJECT_ROOT/src/codex_reproduce_best1123.py" \
  --root "$PROJECT_ROOT" --device cpu
```

CUDA inference is available with `--device cuda` and a suitable CUDA_VISIBLE_DEVICES setting. The tested command writes `reproduction/best1123/g23_greplace.csv` and `inference_verification.json`. It first checks the preserved MLP control and then refuses any mismatch with the actual scored CSV hash.

The command recomputes MLP game embeddings from the saved checkpoint, and recomputes Gaussian scores from the expanded saved LDA model and individual reference-game features. The accepted formula is:

```text
z(cx40 coordinate fusion) + 0.25*z(MLP cosine) + 0.10*z(expanded Gaussian)
```

The older 1,000-player Gaussian contributes **zero**. No expanded cosine score is added. The expanded projection has 128 dimensions and within-class shrinkage .5. Candidate-specific diagonal reference variance is shrunk .8 toward the pooled variance, with a 1e-5 floor. Mean-set uncertainty uses 1/query_games + 1/10. All weights are global; no individual predictions were manually edited.

## Official data and training

Official v1.1 archive SHA-256 is `e0880a3e8723f776e931ecce7fbc09eace9a0dd42b18a1d8ebac4a1628d2395e`. Official training CSVs belong under PROJECT_ROOT/data/training; updated test files under PROJECT_ROOT/data/tests1v2/tests. Parent preprocessing produces data/train_arr.npz.

The MLP uses the original 1,000-player training feature sample. The expanded LDA uses that sample plus two disjoint 2,500-player shards, totaling 90,000 games from 6,000 players, excluding the fixed 800-player holdout. Raw extraction source is codex_policy_expand1122.py; fitting/selection source is codex_policy_expand_fit1122.py. Original and expanded feature arrays, sample indices, logs and models are in the frozen policy releases. HumanSL source/checkpoint and license are preserved in go-policy-residual-pilot-20261119; retain all copyright notices.

Local development/audit rows are recorded in the candidate pack. The replacement variant was fixed before those checks and had only a small local audit gain; its improvement here is the actual competition result. Preserve negative experiments in the dated state.

## Verification scope

The CPU saved-model command was verified byte-exact from frozen coordinate fusion and extracted policy features. The inherited raw-parent base-view reconstruction discrepancy remains documented in prior guides; full raw-input equality is not claimed. Prior strongest releases remain intact. Score receipts and the complete November23 experiment ledger are under go/reproduction/20261123/requests and submission_state.json.
