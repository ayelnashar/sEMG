#!/usr/bin/env bash
# Reproduce every table and figure locally (Linux or macOS).
#   ./run_all.sh            full run, the settings used in the paper (CPU, two threads)
#   QUICK=1 ./run_all.sh    short smoke test with fewer seeds and epochs; numbers will not match the paper
#   USE_GPU=1 ./run_all.sh  allow a GPU; values can differ in the last digits
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$ROOT/src"
export EMG_DATA="${EMG_DATA:-$ROOT/data}"
export EMG_RES="${EMG_RES:-$ROOT/results/run}"
export PYTHONUNBUFFERED=1
export EMG_THREADS="${EMG_THREADS:-2}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" OPENBLAS_NUM_THREADS=1
[ "${USE_GPU:-0}" = "1" ] || export CUDA_VISIBLE_DEVICES=""
RAW="$ROOT/raw"
mkdir -p "$EMG_DATA" "$EMG_RES" "$RAW"

if [ "${QUICK:-0}" = "1" ]; then
  export EPOCHS=3 SEEDS_A=42,43 SEEDS_B=0,1 NS=2 SEEDS_N=42 NPAIRS=2 SEEDS_W=42 SEEDS_WP=42 SEEDS_CI=42,43 NBOOT=100
else
  export EPOCHS=25 SEEDS_A=42,43,44,45,46,47,48,49,50,51 SEEDS_B=0,1,2,3,4,5,6,7,8,9 NS=10 SEEDS_N=42,43,44,45,46 \
         NPAIRS=4 SEEDS_W=42,43,44,45,46 SEEDS_WP=42,43,44 SEEDS_CI=42,43,44,45,46,47,48,49,50,51 NBOOT=2000
fi

# 1. data, pinned to the commits cited in the paper
A_COMMIT=ff82130a8ed2f6838235dfc9f675a157e517e67a
B_COMMIT=3d98c7090b4168c2dd40bf8814217f9dcc4a7f55
A_ZIP_SHA256=28f9c8d8742397d77d9b085be8cc773e2596fc379349a2ed5c2c46a5d79d9c64
fetch() {  # url dest commit
  [ -d "$2/.git" ] || git clone -q --no-checkout "$1" "$2"
  git -C "$2" cat-file -e "$3^{commit}" 2>/dev/null || git -C "$2" fetch -q origin "$3" || git -C "$2" fetch -q --unshallow origin
  git -C "$2" checkout -q --detach "$3"
  [ "$(git -C "$2" rev-parse HEAD)" = "$3" ] || { echo "wrong commit in $2" >&2; exit 1; }
}
fetch https://github.com/Hunter-qing/Forearm-sEMG-Gesture-Dataset.git "$RAW/A" "$A_COMMIT"
fetch https://github.com/aljazfrancic/myo-readings-dataset.git "$RAW/B" "$B_COMMIT"
python - "$RAW/A/dataset.zip" "$A_ZIP_SHA256" <<'PY'
import hashlib, sys
h = hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest()
sys.exit(0 if h == sys.argv[2] else f"dataset.zip checksum mismatch: {h}")
PY
if [ ! -d "$RAW/A_extracted" ]; then
  python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$RAW/A/dataset.zip" "$RAW/A_extracted"
fi
A_PARENT="$(dirname "$(find "$RAW/A_extracted" -maxdepth 3 -type d -name 'subject_001' | head -n 1)")"
check_data() {
  python - "$EMG_DATA" <<'PY'
import sys, numpy as np, os
a = np.load(os.path.join(sys.argv[1], "d1.npz")); b = np.load(os.path.join(sys.argv[1], "d2.npz"))
ok = a["X"].shape == (3000, 1000, 8) and b["X"].shape == (1091, 1000, 8) and list(np.bincount(b["y"])) == [900, 65, 126]
sys.exit(0 if ok else 1)
PY
}
# reparse if the data are missing or were written by the pre-V12 parser (1082 Dataset B windows)
if [ ! -f "$EMG_DATA/d1.npz" ] || [ ! -f "$EMG_DATA/d2.npz" ] || ! check_data; then
  python "$SRC/prepare_data.py" "$A_PARENT" "$RAW/B"
  check_data || { echo "parsed data do not have the expected window counts" >&2; exit 1; }
fi
echo "Data: 3000 Dataset A windows, 1091 Dataset B windows."

# 2. experiments (each skips runs whose result file already exists; use a new EMG_RES after changing code or settings)
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
