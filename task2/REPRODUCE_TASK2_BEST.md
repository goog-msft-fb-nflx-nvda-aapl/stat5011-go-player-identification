# Task 2 best: rank_e13.csv (website 0.7683, 2026-11-22) = e5 recipe (0.7676, 2026-11-21) + 3 extra 350-ply models q41-q43

**e13**: same blend with the seq mean over 14 models: the 11 below plus q41,q42,q43 (`HSL_PLIES=350` = plies 1-350, features of plies 251-350 from `scripts/36_task2_plies251_350_drv4.sh` -> `results/hsl4_*`; `HSL_PLIES=350 SRCS=trainsel,trainsel2,trainsel3,trainsel4 python src/train_hslseq5.py q4X --epochs 80 --d 192 --seed 4X`, X=1,2,3). CPU regeneration: `python src/rank_e5_from_artifacts.py sub/rank_e13.csv artifacts_task2_e5 <test dir> v1,v2,v3,v5,w6,w7,w9,z11,z12,z13,z14,q41,q42,q43` (byte-identical to the uploaded file). Below: e5 recipe details.


Recipe (all log-probabilities, decode = expected-score rule p + e^-1 * neighbours): `1 * mean(bag LMs g1,g2,g3,g4,g6,g7) + 2 * mean(11 HumanSL sequence transformers v1,v2,v3,v5,w6,w7,w9,z11-z14)`. No fine-tuned-game LM and no LightGBM term (grid weight 0).
Runnable script: `scripts/35_task2_best_e5.sh` (environment: `scripts/env.sh`, `requirements.txt`; seeds PYTHONHASHSEED=0 and the per-model `--seed` below).

## Quick verification (CPU, seconds)
`python src/rank_e5_from_artifacts.py sub/rank_e5.csv artifacts_task2_e5 <dir with rank_prediction_test_{public,private}.csv>` regenerates the uploaded file from the 17 saved per-model test log-prob arrays in `artifacts_task2_e5/` (byte-identical to the file that scored 0.7676; `submissions/rank_e5.csv`). Full retraining: `scripts/35_task2_best_e5.sh` (GPU, many hours; GPU nondeterminism means a retrain gives a close, not identical, file).

## Data and external assets
- AILAB-NDHU/aicup2026-go-dataset -> `prep.py` -> data/train_arr.npz. Hold-out = 800 players `np.random.default_rng(1234).choice(np.unique(P),800,replace=False)`.
- External (declared): KataGo b18c384nbt-humanv0 human-SL net (`katago/humanv0.ckpt`), run in PyTorch with KataGo python sources (https://github.com/lightvector/KataGo). 8 rank profiles 20k..8d.
- Move LMs lm1-lm5 (rank-conditioned transformers): `scripts/10_pretrain_lms.sh`, `scripts/22_lm4_lm5_pretrain.sh`.

## Components
1. HumanSL per-move features plies 1-100: `hsl_extract2.py <src> results/hsl2_<src>.npy --maxm 100 --nproc 32` for src in test2 trainsel trainsel2 trainsel3 trainsel4.
2. Plies 101-250: `hsl_extract3.py <src> results/hsl3_<src>_s<k>.npy --maxm 250 --start 100 --nproc 24 --shard k --nshard 3` (k=0,1,2), driver `scripts/34_task2_late_plies_drv3.sh`; `hslcat.py load_full(name)` interleaves the shards and joins with plies 1-100 -> (G,250,27). Pilot: GBM first100 .6217 -> first250 .6534.
3. Bag LMs (`train_ftbag.py`, SORD loss): g1 lm3 ep3 seed0 | g2 lm2 ep4 seed1 | g3 lm3 ep4 seed2 nmax20 np_12 tau1.5 | g4 lm2 ep5 seed3 nmax16 tau1.2 lr1.5e-4 | g6 lm5 ep4 seed6 | g7 lm4 ep4 seed7 (others default: tau1.0 np_16 lr1e-4 nmax12 steps7000 maxlen320); test log-probs via `rank_g1.py <tag>`.
4. HumanSL sequence transformers (`train_hslseq5.py <tag> --epochs 80 --d 192 --seed S`, env `SRCS`): v1 trainsel seed1 | v2,v3,v5 trainsel,trainsel2 seeds 2,3,5 | w6,w7,w9 +trainsel4 seeds 6,7,9 | z11-z14 all four sources seeds 11-14. Hold-out val of the individual models: v1 .704, v2/v3/v5 ~.727, z ~.733-.739.
5. Blend: `blend4567.py` (weights grid) and `mke.py` (key e5).

## Scores
Local hold-out blend: 4 seqs .7429, 7 seqs .7473, 11 seqs .7490 (old 100-ply best .7228). Website: e5 0.7676 (11 seqs), e6 0.7635 (15 seqs incl. y21-y24 d256 100 ep). Not used: u31-u34 (d384, 150 ep; stopped).

## Not backed up
Checkpoints and feature .npy files (server only); regenerate with the script.
