#!/bin/bash
# late-ply (101-250) HumanSL extraction, shard k of 3 (run k=0,1,2 on separate GPUs or sequentially); usage: 34_task2_late_plies_drv3.sh k
SD="$(cd "$(dirname "$0")" && pwd)"; source "$SD/env.sh"; k=$1
export CUDA_VISIBLE_DEVICES=${GPU_BASE:-$k} OMP_NUM_THREADS=4
for s in test2 trainsel trainsel2 trainsel4 trainsel3; do
  [ -f results/hsl3_${s}_s$k.npy ] || $PY src/hsl_extract3.py $s results/hsl3_${s}_s$k.npy --maxm 250 --start 100 --nproc 24 --shard $k --nshard 3 > logs/hsl3_${s}_s$k.log 2>&1
done
