# Task 1 best on November 22: 0.8686865

`g22_cx40.csv` scored **0.8686865**. SHA-256 `123a07fc353790bc983a76862d10db96e03399df4160a3e7af44deb293b61fdb`. Exact CSV is `go/submissions/g22_cx40.csv`; receipt and all five outcomes are under `go/runs/` and `go/reproduction/20261122/submission_state.json`. The later LM6 test scored .8682102, so preserve this earlier best for the user's manual November24 final upload.

## Restore official data, environment and models

Follow `REPRODUCE_TASK1_BEST_20261121.md` for the pinned Python3.11/PyTorch2.6 environment, official dataset paths, frozen parent, HumanSL feature extraction, MLP model and LDA model. Official v1.1 archive SHA-256 is `e0880a3e8723f776e931ecce7fbc09eace9a0dd42b18a1d8ebac4a1628d2395e`. Training CSVs belong under PROJECT_ROOT/data/training and official test CSVs under PROJECT_ROOT/data/tests1v2/tests. External HumanSL source/model/license are preserved in the policy pilot release; retain all notices.

Additionally restore private release `go-next-experiments-20261119` assets `cx40_lm4_noaug_all_training.tar.gz` and `cx40_lm4_noaug_all_inference.tar.gz` preserving paths. The former contains the selected LM4-derived player encoder and training source/settings; the latter contains its inference source/settings and fusion results. Required score arrays are `reproduction/experiments_1119/runs/cx40_lm4_noaug_all/results/Sfinal_newmember_candidate_{pub2,priv2}.npy`.

The selected encoder was predefined epoch2 of a four-epoch run: LM4 initialization, seed0, learning rate .0001, ArcFace scale20/margin .1, all training identities, no orientation augmentation. Exact commands are in `go/reproduction/20261119/next_experiment_jobs.json`. Model hash, selected checkpoint and frozen training dependencies are recorded in the cx40 training archive.

## Reproduce predictions from saved models

Copy this release's go/src files into PROJECT_ROOT/src. Restore all Nov21 dependencies, plus the cx40 fusion above, then run:

```bash
export PROJECT_ROOT=/absolute/path/to/restored/project
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python "$PROJECT_ROOT/src/codex_reproduce_best1122.py" --root "$PROJECT_ROOT" --device cuda
```

Output is `reproduction/best1122/g22_cx40.csv`. It verifies the previous policy control, recomputes MLP game embeddings and the shrunk Gaussian score from the saved LDA model and individual reference games, then applies `z(cx40 fusion) + .25*z(MLP cosine) + .1*z(Gaussian)`. It refuses a mismatch with the scored CSV hash and saves inference_verification.json. Original-family rows remain inherited from the accepted parent.

To regenerate the cx40 fusion from its encoder checkpoint, restore the full cx40 inference workspace and parent dependencies and run:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 python "$PROJECT_ROOT/src/codex_new_member_nov_1119.py" --root "$PROJECT_ROOT/reproduction/experiments_1119/runs/cx40_lm4_noaug_all" --mode inference
```

The run workspace's reproduction/new_member_config.json, src files, selected model, parent arrays and official test paths must be restored from the frozen archives first. This command asserts the inherited parent control before inferring and fusing the new encoder.

## Compact cached replay and verification scope

Private release `go-daily-candidates-20261122` / `daily_candidates1122.tar.gz` preserves all score inputs with hashes, per-candidate settings, source and predictions. Running `codex_daily_candidates1122.py --root PROJECT_ROOT` with an absent output directory reproduces these candidates and asserts the Nov21 control. No per-question predictions are manually edited.

Saved-model policy inference from frozen cx40 fusion and extracted HumanSL features is verified byte-exact. The inherited raw-parent base-view mismatch remains documented in the Nov21 guide. A separate raw reconstruction was evaluated at .8670060 and preserved; it is not the accepted best. Full raw-data equality of the accepted parent remains unresolved.
