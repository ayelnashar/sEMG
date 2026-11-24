#!/usr/bin/env bash
# Reproduce every table and figure locally (Linux or macOS).
#   ./run_all.sh            full run, the settings used in the paper
#   QUICK=1 ./run_all.sh    short smoke test with fewer seeds and epochs
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$ROOT/src"
export EMG_DATA="${EMG_DATA:-$ROOT/data}"
export EMG_RES="${EMG_RES:-$ROOT/results/run}"
export PYTHONUNBUFFERED=1
RAW="$ROOT/raw"
mkdir -p "$EMG_DATA" "$EMG_RES" "$RAW"

if [ "${QUICK:-0}" = "1" ]; then
  export EPOCHS=3; SEEDS_A=42,43; SEEDS_B=0,1; export NS=2; SEEDS_N=42; export NPAIRS=2
  SEEDS_W=42; SEEDS_WP=42; SEEDS_CI=42,43; export NBOOT=100
else
  export EPOCHS=25; SEEDS_A=42,43,44,45,46,47,48,49,50,51; SEEDS_B=0,1,2,3,4,5,6,7,8,9; export NS=10
  SEEDS_N=42,43,44,45,46; export NPAIRS=4; SEEDS_W=42,43,44,45,46; SEEDS_WP=42,43,44
  SEEDS_CI=42,43,44,45,46,47,48,49,50,51; export NBOOT=1000
fi
export SEEDS_B

# 1. data
[ -d "$RAW/A" ] || git clone -q --depth 1 https://github.com/Hunter-qing/Forearm-sEMG-Gesture-Dataset.git "$RAW/A"
[ -d "$RAW/B" ] || git clone -q --depth 1 https://github.com/aljazfrancic/myo-readings-dataset.git "$RAW/B"
if [ ! -d "$RAW/A_extracted" ]; then
  python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$RAW/A/dataset.zip" "$RAW/A_extracted"
fi
A_PARENT="$(dirname "$(find "$RAW/A_extracted" -maxdepth 3 -type d -name 'subject_*' | sort | head -n 1)")"
if [ ! -f "$EMG_DATA/d1.npz" ] || [ ! -f "$EMG_DATA/d2.npz" ]; then
  python "$SRC/prepare_data.py" "$A_PARENT" "$RAW/B"
fi

# 2. experiments (each skips runs whose result file already exists)
cd "$SRC"
SEEDS=$SEEDS_A  python v3_a.py
SEEDS=$SEEDS_B  python v3_b.py
python v3_splits.py
SEEDS=$SEEDS_N  python v3_nov.py
SEEDS=$SEEDS_W  python v3_200.py
python v3_lda.py
SEEDS=$SEEDS_WP python v3_within.py
SEEDS=$SEEDS_CI python v3_ci.py

# 3. aggregation, tables, figures
python v3_agg.py
python v3_tables.py
python v3_figs.py
echo "Done. Tables in $EMG_RES/tables, figures in $EMG_RES/figures."
