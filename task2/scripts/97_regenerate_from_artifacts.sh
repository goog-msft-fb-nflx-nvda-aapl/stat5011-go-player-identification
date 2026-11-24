#!/bin/bash
# NO-GPU verification: regenerate both best submissions byte-identically from the saved score matrices / test log-probs.
# Also unpack t1t2_extra_artifacts.tar.gz (Sidft_c11/c12). Needs: GitHub release artifacts-2026-11-16 of this repo (small_artifacts.tar.gz = Task 1 matrices, t2_artifacts.tar.gz = Task 2 components), unpacked inside $GO_ROOT, and the AIdea test csv in $GO_ROOT/data.
SD="$(cd "$(dirname "$0")" && pwd)"; source "$SD/env.sh"; cp "$SD"/../src/*.py src/
cp data/tests1v2/tests/sample_submission_player_v2.csv sub/base.csv
# Task 1: ensemble of per-model score matrices (weights = repeats) + gate/CSLS/Sinkhorn fusion -> id_v2_idftJ.csv
$PY src/ens_sidft.py ens7 a2 b1 b1 c1 c1 c3 c3 c2 b2 b3 c4 c4 c5 c5 c6 c6 c7 c8 c8 c8 c9 c9 c10 c10 c10 c11 c11 c12
$PY src/mk_gate5.py sub/base.csv sub/id_v2_idftJ.csv 2 3 1.4 1.0 ens7 3 1 0.6 0.5 0.5 a2,b1,b2,b3,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12 0   # needs t1_qemb_artifacts.tar.gz too
# Task 2: HumanSL LightGBM + ft45 + bag-LM + HumanSL-sequence log-prob blend -> rank_hsl7.csv
$PY src/rank_final_from_artifacts.py sub/rank_hsl7.csv
bash "$SD/98_verify.sh"      # prints agreement with submissions/*.csv (expect 1.0000: both were verified byte-identical on 2026-11-16)
# Task 2 BEST rank_e5.csv (website 0.7676) from the 17 saved per-model test log-prob files (artifacts_task2_e5/, in this repo; byte-identical to the uploaded file, verified 2026-11-21)
$PY src/rank_e5_from_artifacts.py sub/rank_e5.csv "$SD/../artifacts_task2_e5"
