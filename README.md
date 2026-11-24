# Confidence and Density Rejection Under Distribution Shift in Surface Electromyography

Simulation and analysis code for the manuscript of the same title by Ayman Elnashar (under review). It reproduces every table and figure in the paper from two public eight-channel armband datasets.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ayelnashar/sEMG/blob/main/notebooks/Reproduction_V11.ipynb)

## What the study does

A self-supervised transformer encoder (371 584 parameters) is pretrained by masked patch reconstruction on surface EMG, and a temperature-calibrated linear probe classifies gestures on its frozen representation. Five abstention statistics (Mahalanobis density, maximum softmax probability, entropy, margin and energy), a deterministic range check and a classical time-domain feature pipeline with linear discriminant analysis are evaluated under:

- a shift ladder: held-out session, new participants, a second dataset, and conditioning mismatch (signed input to a model trained on rectified input);
- the same mismatch repeated inside one dataset, in both directions and with training on both conventions;
- further mismatches in gain, envelope smoothing, band-pass filtering, per-window standardization and sampling rate;
- five simulated sensor faults and held-out gestures excluded from pretraining;
- 200 ms decision windows;
- a mixed-fault experiment that combines the range check, density gate and confidence threshold;
- the encoder depth at which the density score is computed.

## Data

Both datasets are downloaded from their public repositories by the notebook or `run_all.sh`; nothing is redistributed here.

| | Dataset | Participants | Conditioning | License |
|---|---|---|---|---|
| A | [Forearm sEMG Gesture Dataset](https://github.com/Hunter-qing/Forearm-sEMG-Gesture-Dataset) | 30, five sessions | stored rectified, 0 to 128 | CC BY 4.0 |
| B | [Myo readings dataset](https://github.com/aljazfrancic/myo-readings-dataset) | 38 | signed bytes, −128 to 127 | MIT |

`src/prepare_data.py` parses them into 1000-sample, eight-channel windows: 3000 windows for Dataset A and 1082 for Dataset B.

## Running it

**Google Colab.** Open the notebook with the badge above and choose Runtime > Run all. It works on CPU or GPU. With `SAVE_TO_DRIVE = True` results are kept on Google Drive and a disconnected run resumes where it stopped. Set `QUICK = True` for a smoke test of about 15 minutes.

**Locally.**

```bash
pip install -r requirements.txt
./run_all.sh              # settings used in the paper
QUICK=1 ./run_all.sh      # smoke test, numbers will not match the paper
```

Results go to `results/run/`, with tables as CSV in `results/run/tables/` and figures in `results/run/figures/`. Every experiment writes one file per seed or split and skips files that already exist, so an interrupted run can be restarted.

**Runtime.** About two hours on two CPU cores, and considerably less on a GPU.

## Code

| File | Purpose | Paper |
|---|---|---|
| `src/prepare_data.py` | parse both datasets | Section III-A, Table II |
| `src/core.py` | encoder, pretraining, density gate, sensor faults, metrics | Sections III-B to III-G |
| `src/lib2.py` | conditioning mismatches, range check, time-domain features with LDA | Sections III-C, III-F |
| `src/lib3.py` | calibrated linear probe and abstention statistics | Sections III-E, III-F |
| `src/v3_a.py` | Dataset A, ten seeds: ladder, faults, mismatch family, error ranking, mixed faults, normalization factorial | Tables III, IV, VI, VIII, X, XI; Figs. 2 to 8 |
| `src/v3_b.py` | within Dataset B, ten participant splits | Tables V, VI; Fig. 3 |
| `src/v3_splits.py` | ten participant splits of Dataset A, encoder depth | Table IX; Figs. 3, 9, 10 |
| `src/v3_nov.py` | held-out gestures excluded from pretraining | Table IV |
| `src/v3_200.py` | 200 ms windows with an eight-token encoder | Fig. 8 |
| `src/v3_lda.py` | classical time-domain features with LDA | Table VII |
| `src/v3_within.py` | within-participant control | Section IV-L |
| `src/v3_ci.py` | participant-clustered bootstrap intervals | Sections IV-A, IV-B |
| `src/v3_agg.py` | aggregation and statistical tests | Table XII |
| `src/v3_tables.py`, `src/v3_figs.py` | CSV tables and figures | all |

## Reference results

`results/reference/` holds the tables and figures produced by the full CPU run reported in the paper. A CPU run with the settings of `run_all.sh` reproduced them exactly in our environment (PyTorch 2.15.0, NumPy 2.5.3, scikit-learn 1.9.1); other library versions can shift the last digit. A GPU run reproduces the findings, but individual values can differ in the third decimal because reduction order differs between processors; TF32 is disabled so that GPU arithmetic stays in full single precision.

## Citation

If you use this code, please cite the manuscript:

> A. Elnashar, "Confidence and Density Rejection Under Distribution Shift in Surface Electromyography," manuscript under review, 2026.
