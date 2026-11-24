# Task 2 — Nov 23 experiments (candidates f1–f4, for manual upload Nov 24)
Base (best on website): e13 = 1*mean(bag LMs g1,g2,g3,g4,g6,g7) + 2*mean(14 Transformer bag-pooled HumanSL sequence models) (.7683).
New code: `src/train_hslseq6.py` (adds --arch gru|tf, --nmax, --bagattn), `src/mkf.py` (builds blended csv), `hsl_extract3.py` source `trainsel5` (+138,204 games, up to 40 unused games/non-hold-out player, rng seed 55, KataGo HumanSL plies 1-250, 3 shards), `hslcat.py` loader.
Extraction: `python src/hsl_extract3.py trainsel5 results/hsl5_trainsel5_s$k.npy --maxm 250 --start 0 --nproc 24 --shard $k --nshard 3` (k=0,1,2); then idx = interleave of shard idx files (idx[k::3]=shard k) saved as results/hsl2_trainsel5_idx.npy.
Training: `SRCS=<sources> HSL_PLIES=250 python src/train_hslseq6.py <tag> --epochs 80 --d 192 --seed S [--arch gru]`.
- GRU r62,r64,r65,r66,r67 (old 4 sources, seeds 62,64-67; r62 val .7451)
- GRU s71-s74 (SRCS=trainsel..trainsel5, seeds 71-74, val .744-.747); TF s75-s78 (seeds 75-78, val .736-.746)
Candidates (O = 14 TF v1,v2,v3,v5,w6,w7,w9,z11,z12,z13,z14,q41,q42,q43; G = r62,r64-r67; S = s71..s78):
- f1: `mkf.py f1 1 2 O,G` (97.75% agree with e13)
- f2: `mkf.py f2 1 2 S` (93.4%)
- f3: `mkf.py f3 1 2 O,G,S` (96.4%)
- f4: `mkf.py f4 1.5 1 O,G S 2` (94.9%)
Local hold-out (800 players) all ≈ .750-.751 (saturated; differences < .005 unreliable). Judged best a priori: f3 (largest ensemble, most data).
