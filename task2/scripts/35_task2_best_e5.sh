#!/bin/bash
# Task 2 BEST: sub/rank_e5.csv (website 0.7676, rank 12/161, uploaded 2026-11-21; local held-out blend 0.7490)
#   logP = 1*mean(bag-level LMs g1,g2,g3,g4,g6,g7 test log-probs) + 2*mean(11 HumanSL(plies 1-250) sequence transformers v1,v2,v3,v5,w6,w7,w9,z11,z12,z13,z14); decode = expected-score rule (src/mke.py key e5)
# Prerequisites: scripts/00_setup.sh (env), 10_pretrain_lms.sh (lm1-lm3), 22_lm4_lm5_pretrain.sh (lm4, lm5), KataGo humanv0.ckpt at $GO_ROOT/katago/humanv0.ckpt + KataGo python sources at $GO_ROOT/katago/kgsrc (https://github.com/lightvector/KataGo, b18c384nbt-humanv0 human SL net)
# hold-out = 800 players, np.random.default_rng(1234).choice(np.unique(P),800,replace=False) (identical for every component)
set -e; source "$(dirname "$0")/env.sh"; export CUDA_VISIBLE_DEVICES=$GPU
# 1) HumanSL features, plies 1-100 (8 rank profiles, per-move logp/entropy/top1/win/lead): ~8 h per source on one GPU; trainsel=3 sources+hold, trainsel3/4 = extra games per player
for s in test2 trainsel trainsel2 trainsel3 trainsel4; do $PY src/hsl_extract2.py $s results/hsl2_$s.npy --maxm 100 --nproc 32 > logs/hsl2_$s.log 2>&1; done
# 2) HumanSL features, plies 101-250 (3 shards; run k=0,1,2 on three GPUs: scripts/34_task2_late_plies_drv3.sh k)
for k in 0 1 2; do GPU_BASE=$GPU bash "$(dirname "$0")/34_task2_late_plies_drv3.sh" $k; done   # shards are independent: run in parallel on 3 GPUs (GPU_BASE=<idx>) to save time
# 3) bag-level fine-tuned LMs (SORD loss, bags of 5..nmax games of one player); exact args read from the checkpoints
$PY -u src/train_ftbag.py g1 --init lm3 --epochs 3                                           > logs/ftbag_g1.log 2>&1
$PY -u src/train_ftbag.py g2 --init lm2 --epochs 4 --seed 1                                  > logs/ftbag_g2.log 2>&1
$PY -u src/train_ftbag.py g3 --init lm3 --epochs 4 --seed 2 --nmax 20 --np_ 12 --tau 1.5     > logs/ftbag_g3.log 2>&1
$PY -u src/train_ftbag.py g4 --init lm2 --epochs 5 --seed 3 --nmax 16 --tau 1.2 --lr 1.5e-4  > logs/ftbag_g4.log 2>&1
$PY -u src/train_ftbag.py g6 --init lm5 --epochs 4 --seed 6                                  > logs/ftbag_g6.log 2>&1
$PY -u src/train_ftbag.py g7 --init lm4 --epochs 4 --seed 7                                  > logs/ftbag_g7.log 2>&1
for t in g1 g2 g3 g4 g6 g7; do $PY src/rank_g1.py $t > logs/rank_$t.log 2>&1; done            # results/ftbag_<t>_test_lp.npy (+ holdfeat)
$PY src/hsl_rank3b.py > logs/hsl_rank3b.log 2>&1                                              # results/hslr3_val.npz (hold-out bags, needed for local blend check)
# 4) 250-ply HumanSL sequence transformers (train_hslseq5.py; hslcat.py joins plies 1-100 + 101-250; bag-pooled, SORD; also writes valLP/testLP)
export OMP_NUM_THREADS=4
SRCS=trainsel                                        $PY -u src/train_hslseq5.py v1  --epochs 80 --d 192 --seed 1  > logs/hslseq5_v1.log  2>&1
SRCS=trainsel,trainsel2                              $PY -u src/train_hslseq5.py v2  --epochs 80 --d 192 --seed 2  > logs/hslseq5_v2.log  2>&1
SRCS=trainsel,trainsel2                              $PY -u src/train_hslseq5.py v3  --epochs 80 --d 192 --seed 3  > logs/hslseq5_v3.log  2>&1
SRCS=trainsel,trainsel2                              $PY -u src/train_hslseq5.py v5  --epochs 80 --d 192 --seed 5  > logs/hslseq5_v5.log  2>&1
SRCS=trainsel,trainsel2,trainsel4                    $PY -u src/train_hslseq5.py w6  --epochs 80 --d 192 --seed 6  > logs/hslseq5_w6.log  2>&1
SRCS=trainsel,trainsel2,trainsel4                    $PY -u src/train_hslseq5.py w7  --epochs 80 --d 192 --seed 7  > logs/hslseq5_w7.log  2>&1
SRCS=trainsel,trainsel2,trainsel4                    $PY -u src/train_hslseq5.py w9  --epochs 80 --d 192 --seed 9  > logs/hslseq5_w9.log  2>&1
for s in 11 12 13 14; do SRCS=trainsel,trainsel2,trainsel3,trainsel4 $PY -u src/train_hslseq5.py z$s --epochs 80 --d 192 --seed $s > logs/hslseq5_z$s.log 2>&1; done
# 5) local blend check (hold-out) and final file
$PY src/blend4567.py g1,g2,g3,g4,g6,g7 v1,v2,v3,v5,w6,w7,w9,z11,z12,z13,z14    # best (ft,gbm,bag,seq) weights (0,0,1,2) -> local ~0.749
$PY src/mke.py                                                                  # writes sub/rank_e1..e6.csv; e5 = 1*bag + 2*mean(11 seq) ; e6 adds y21-y24 (d256, 100 ep, seeds 21-24) and scored 0.7635
# --- e13 (website 0.7683, best): + 350-ply models. plies 251-350 features, then 3 more seeds:
for k in 0 1 2; do GPU_BASE=$GPU bash "$(dirname "$0")/36_task2_plies251_350_drv4.sh" $k; done
for s in 41 42 43; do HSL_PLIES=350 SRCS=trainsel,trainsel2,trainsel3,trainsel4 $PY -u src/train_hslseq5.py q$s --epochs 80 --d 192 --seed $s > logs/hslseq5_q$s.log 2>&1; done
$PY src/mke.py      # key e13 = 1*bag + 2*mean(v1,v2,v3,v5,w6,w7,w9,z11-z14,q41,q42,q43)
