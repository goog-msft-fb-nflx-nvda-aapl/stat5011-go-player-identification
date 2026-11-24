#!/bin/bash
# plies 251-350 HumanSL extraction, shard k of 3; usage: 36_task2_plies251_350_drv4.sh k  (env: GO_ROOT PY GPU_BASE)
k=$1; cd ${GO_ROOT:-/home/jtan/solardefect/go_20261113}; P=${PY:-/home/jtan/solardefect/codex_20261028/venv/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU_BASE:-$k} OMP_NUM_THREADS=4
for s in test2 trainsel trainsel2 trainsel4 trainsel3; do
  [ -f results/hsl4_${s}_s$k.npy ] || $P src/hsl_extract3.py $s results/hsl4_${s}_s$k.npy --maxm 350 --start 250 --nproc 24 --shard $k --nshard 3 > logs/hsl4_${s}_s$k.log 2>&1
done
