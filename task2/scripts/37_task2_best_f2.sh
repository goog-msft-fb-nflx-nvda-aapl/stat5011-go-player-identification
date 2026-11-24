#!/bin/bash
# Task 2 BEST: submissions/rank_f2.csv (website 0.7691850, rank 17/171, uploaded 2026-11-24)
#   logP = 1*mean(bag-level fine-tuned LMs g1,g2,g3,g4,g6,g7) + 2*mean(8 HumanSL(plies 1-250) bag-pooled sequence models s71..s78 trained on trainsel..trainsel5); decode = expected-score rule
# Prerequisites: scripts/35_task2_best_e5.sh steps 1-3 (features plies 1-100 and 101-250 for test2,trainsel..trainsel4; bag LMs g1..g6,g7 and their test log-probs), env.sh, KataGo humanv0 net.
# Fast verification without GPU: python src/rank_e5_from_artifacts.py out.csv artifacts_task2_f2 <test dir> s71,s72,s73,s74,s75,s76,s77,s78  (byte-identical to submissions/rank_f2.csv, sha256 2926dbf149c255f4df47e00a046f3ecbad64c09c7ee2ecc93e678da23cd2ef98)
set -e; source "$(dirname "$0")/env.sh"
# 1) NEW games: trainsel5 = up to 40 games/player (rng seed 55) not in trainsel..trainsel4, non-hold-out players only -> 138,204 games; plies 1-250 in one pass, 3 shards (~5.3 h each on one GPU; run shards in parallel on 3 GPUs)
for k in 0 1 2; do CUDA_VISIBLE_DEVICES=$k OMP_NUM_THREADS=4 $PY src/hsl_extract3.py trainsel5 results/hsl5_trainsel5_s$k.npy --maxm 250 --start 0 --nproc 24 --shard $k --nshard 3 > logs/hsl5_trainsel5_s$k.log 2>&1 & done; wait
$PY src/mk_trainsel5_idx.py     # -> results/hsl2_trainsel5_idx.npy (138204); hslcat.load_full('trainsel5') reads the shards directly
# 2) 8 sequence models (src/train_hslseq6.py; --arch gru = 2-layer bidirectional GRU, default tf = 3-layer Transformer d192; 80 epochs; bag-pooled, SORD head); SRCS = 5 sources, HSL_PLIES=250
export SRCS=trainsel,trainsel2,trainsel3,trainsel4,trainsel5 HSL_PLIES=250 OMP_NUM_THREADS=4
for s in 71 72 73 74; do CUDA_VISIBLE_DEVICES=$((s-71)) $PY -u src/train_hslseq6.py s$s --epochs 80 --d 192 --seed $s --arch gru > logs/hslseq6_s$s.log 2>&1 & done; wait
for s in 75 76 77 78; do CUDA_VISIBLE_DEVICES=$((s-75)) $PY -u src/train_hslseq6.py s$s --epochs 80 --d 192 --seed $s > logs/hslseq6_s$s.log 2>&1 & done; wait
# 3) final file: log P = 1*bag + 2*mean(s71..s78); needs data/tests2/tests/rank_prediction_test_{public,private}.csv
$PY src/mkf.py f2 1 2 s71,s72,s73,s74,s75,s76,s77,s78     # -> sub/rank_f2.csv
# (other candidates: f1 = 1 2 O,G ; f3 = 1 2 O,G,S ; f4 = 1.5 1 O,G S 2 ; O = 14 TF of e13, G = GRU r62,r64-r67 (train_hslseq6.py --arch gru, 4 sources, seeds 62,64,65,66,67))
