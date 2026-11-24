# Task 1 reproduction

The accepted highest public score is **0.8721950**, `g23_greplace.csv`, byte-identical to the last restored `best_872195.csv`. The exact file is `results/best_prediction.csv`; its SHA-256 is:

```
eac2f9120aff4dd74102d08b2cbcc3e2cf9009d976fd80a63890b371deacde26
```

## Official inputs and raw extraction

The organizer supplies training CSVs under `PROJECT_ROOT/data/training` and updated original/new candidate/query CSVs under `PROJECT_ROOT/data/tests1v2/tests`. `model_src/prep.py` is part of the archived competition source; `GO_ROOT=PROJECT_ROOT SUBSAMPLE=1 python src/prep.py` regenerates the training cache. Raw datasets and credentials are not bundled. Follow `FROZEN_BEST_GUIDE.md` for exact source/configuration, pinned KataGo source/model/license, extraction, training, parent dependencies and checksums. The accepted expanded projection is at `reproduction/policy_expanded_lda1122/policy_lda.npz`, replacing the older Gaussian projection.

## Consolidated clean-workspace replay

Authorized graders can receive the compact private release directly or download it from:
https://github.com/goog-msft-fb-nflx-nvda-aapl/aicup2026-solar-defect/releases/tag/go-organizer-package-20261124

The archive is `organizer_saved_model_replay1124.tar.gz`, SHA-256 `b0b6f144a3c0961566472f680f72112e59df236f273a69aba60af57007f404c8`.

```bash
python3.11 -m venv .model-venv
. .model-venv/bin/activate
python -m pip install -r docs/organizer_task1/requirements-replay.txt
python docs/organizer_task1/reproduce.py \
  --archive /absolute/path/to/organizer_saved_model_replay1124.tar.gz \
  --workspace /absolute/path/to/EMPTY-workspace
```

The runner validates all22 input hashes and regenerates MLP embeddings and expanded Gaussian matching on CPU. Output is `EMPTY-workspace/reproduction/best1123/g23_greplace.csv`; it rejects a different hash. The archive includes no official CSVs. It **does** contain declared derived policy features, frozen coordinate-fusion scores and inherited original-family predictions. These are not presented as raw-data inference or trained weights. `results/best_inference_verification.json` records the passed empty-workspace replay using the recorded server environment; a newly installed dependency environment was not separately tested.

For the full competition restoration chain see `FROZEN_BEST_GUIDE.md` and the backing source at the organizer release tag. The updated entry point `model_src/codex_reproduce_best1123.py` uses the same preserved MLP and an expanded LDA learned from90,000 games/6,000 identities. Final score is `z(cx40) + .25*z(MLP cosine) + .10*z(expanded Gaussian)`. Expanded cosine and the older Gaussian have zero final weights.

## Verification boundary

Saved-model replay is byte-exact. The inherited legacy base-view component still changes lower-ranked predictions when rebuilt from raw official CSVs. Full raw-data reconstruction is unresolved. A separate raw-input alternative scored.8670060 and is not the accepted best. Source availability and a report do not alone establish compliance with an organizer's full reproducibility requirement.

The submission period has ended. The course copy reports the dated public best, not a verified final private result. Task2 reproduction is independently preserved in `task2/REPRODUCE_TASK2_BEST.md`.
