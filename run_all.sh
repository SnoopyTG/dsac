#!/usr/bin/env bash
# Runs Phases 1-3 end to end. Usage: bash run_all.sh [quick|full]
set -e
MODE=${1:-quick}
RIDERS=20; CV=5; [ "$MODE" = "full" ] && RIDERS=40 && CV=5

echo "== Tests ==";            python test_pipeline.py
echo "== Phase 1: generate =="; python data_generator.py --out data --riders $RIDERS --fs 50 --duration 300 --validate --plot
echo "== Phase 2: features =="; python feature_extraction.py --data data/synthetic_imu.csv --out data --win 30 --hop 15 --w_lean 0.0
python check_features.py --data data
echo "== Phase 3: baselines =="
python apla_dtw_baseline.py --data data --name baseline_relr
python apla_dtw_baseline.py --data data --name baseline_fixedr --fixed_r 1
echo "== Phase 3: CNN single split + ablations =="
for AUX in none hs all_no_time all; do python train.py --data data --aux $AUX --name cnn_$AUX; done
echo "== Phase 3: leave-riders-out CV (the numbers to report) =="
python apla_dtw_baseline.py --data data --name baseline_fixedr_cv --fixed_r 1 --cv $CV
for AUX in hs all; do python train.py --data data --aux $AUX --cv $CV --name cnn_${AUX}_cv; done
echo "== Summary =="; python compare_results.py runs
