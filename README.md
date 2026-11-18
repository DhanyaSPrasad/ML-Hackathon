# PathSnap — Distributed Map-Matching & Route Reconstruction

PathSnap reconstructs plausible directed-road sequences from noisy GPS observations. The repository contains the data-processing and feature-engineering pipeline, five-model experiment, Optuna Bayesian search, HMM/Viterbi decoder, FastAPI prediction service, Leaflet UI and visualisations.

## Dataset layout and partition safety

The code supports both layouts used by the supplied material:

```text
Student folder/
├── train/                 # 80 training trips
├── student_test/          # 10 development trips (may be nested under train/)
└── evaluator_test/        # 10 final-evaluation trips (may be nested under train/)
```

A trip folder contains `<trip_id>.track`, `.nodes`, `.arcs` and `.route` files. The partition resolver checks both `Student folder/<partition>` and `Student folder/train/<partition>`.

**Partition rules implemented in code:**
- `scripts/preprocess.py`, `scripts/train.py`, `scripts/tune.py`, `scripts/evaluate.py` and `scripts/visualize.py` use only `train` and/or `student_test`.
- No training, feature selection, hyperparameter selection or development metric is computed on `evaluator_test`.
- The prediction API first calls `read_trip(..., include_route=False)` and generates candidates with `route=None`. Only after prediction is complete, it may read the reference `.route` file to calculate server-side Top-1 accuracy and candidate miss rate. Raw reference labels never enter features, model inputs, decoding, or the API response; the UI receives only aggregate metrics.
- For the final demonstration, set `PATHSNAP_TRIP_PARTITION=evaluator_test` so the UI lists the ten evaluator trips. Prediction reads `.track`, `.nodes` and `.arcs`; reference `.route` is read only after prediction to calculate aggregate evaluation metrics.

Do not use evaluator reference labels for development, tuning or model selection.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Python 3.10+ is supported. Use the same environment for all scripts. For a quick tuning smoke test, set `$env:PATHSNAP_TUNE_TRIALS="2"` and `$env:PATHSNAP_DECODER_TRIALS="2"`. Remove these overrides for the full experiment (defaults: 60 classifier trials and 60 decoder trials). Training and inference always use the complete original GPS tracks.

## Reproducible experiment workflow

Run from the repository root:

```powershell
python scripts/preprocess.py
python scripts/explore.py
python scripts/train.py
python scripts/tune.py
python scripts/tune_decoder.py
python scripts/evaluate.py
python scripts/visualize.py
python -m unittest discover -s tests -v
```

After the model and decoder settings are finalized, run `python scripts/evaluate_final.py` once to record the official `evaluator_test` inference metrics. Do not use those metrics to tune or select models.

### What each stage does

1. **Preprocessing / audit** (`scripts/preprocess.py`, `pathsnap/core.py`): parses whitespace-delimited files, coerces numeric fields, removes invalid GPS rows, sorts observations chronologically without inventing interpolated positions, validates directed arc endpoints and records missing/invalid rows, timestamps, gaps, geography and network statistics.
2. **Feature engineering** (`pathsnap/core.py`): builds a local metric coordinate plane; indexes directed-road midpoints with a KD-tree; uses a complete in-radius segment shortlist plus a nearest-midpoint fallback, ranks candidates by exact point-to-segment distance, and computes geometric, motion and circular-direction features. Training, validation and API inference all retain the original full-resolution GPS sequence; temporal features are never calculated on downsampled tracks.
3. **Five-model experiment** (`scripts/train.py`): Logistic Regression, KNN, Decision Tree, Random Forest and Extra Trees; median imputation is fitted inside each model pipeline and scaling is applied to LR/KNN. Uses 5-fold `GroupKFold` with trip ID as group and selects the model by mean observation-level Top-1 accuracy. Candidate groups with no positive candidate are excluded from classifier fitting, but remain included in observation-level evaluation so candidate misses count as errors. The student-test partition is reported separately and is never used to select the model. Confusion matrices and tree feature-importance plots are saved.
4. **Classifier Bayesian optimisation** (`scripts/tune.py`): Optuna TPE search for the classifier selected by training-only 5-fold trip-grouped observation-level Top-1 accuracy. Saves trial history, best configuration and tuned model.
5. **Decoder Bayesian optimisation** (`scripts/tune_decoder.py`): generates out-of-fold predictions for training trips and tunes HMM transition penalty, gap reset, movement feasibility and emission distance scale. Student-test and evaluator labels are not used.
6. **Sequential decoding** (`scripts/evaluate.py`): compares classifier-only predictions with movement-aware Viterbi decoding on student-test trips. Repeated observations on the same arc are valid transitions; a long timestamp gap relaxes continuity constraints.
7. **Visualisation** (`scripts/visualize.py`): creates actual-result comparison plots and an interactive Folium map.

## Feature and label definitions

The classifier is a candidate scorer: each GPS observation has eight directed-road candidates. A KD-tree midpoint index is used as a broad-phase search, augmented by a radius that is large enough to include every segment within the configured GPS radius; candidates are then ranked by exact point-to-segment distance. The positive label is assigned by monotonic dynamic-programming alignment against the ordered reference route, with a small heading cost when GPS movement is observable. This remains a **weak geometric label**, not independently verified ground truth. Candidate misses are measured separately; observations whose true target is absent are excluded from binary classifier fitting rather than being mislabeled as all-negative examples.

Reference routes are used in offline training/development scripts and, for final evaluator requests, loaded only after prediction to calculate aggregate server-side metrics. They never enter the inference feature matrix or decoder. Distances use a trip-local equirectangular metric approximation, suitable for the small spatial extent of an individual trip.

**Primary metric:** observation-level Top-1 accuracy selects one candidate per GPS observation and compares its arc ID with the weak reference target. Candidate-level binary metrics are secondary diagnostics. Use trip-grouped CV for model selection and student-test only as a separate development check. Both classifier and decoder TPE searches optimize observation-level Top-1 using training-only folds.

## Key outputs

| Output | Location |
|---|---|
| Preprocessing audit | `pathsnap/results/preprocessing_audit.csv` |
| Feature distributions | `pathsnap/results/feature_summary_by_trip.csv` |
| Five-model comparison | `pathsnap/results/model_comparison.csv` |
| Selected model and feature metadata | `pathsnap/models/model_selection.json`, `model_metadata.json` |
| Five-model comparison | `pathsnap/results/model_comparison.csv` |
| Confusion matrices | `pathsnap/visuals/confusion_*.png` |
| Tree feature importance | `pathsnap/results/feature_importance_*.csv` |
| Classifier Bayesian trials / best parameters | `pathsnap/results/bayesian_trials.csv`, `bayesian_best.json` |
| Decoder Bayesian trials / best parameters | `pathsnap/results/decoder_trials.csv`, `pathsnap/models/decoder_metadata.json` |
| Tuned vs untuned comparison | `pathsnap/results/tuned_vs_untuned.csv` |
| HMM comparison | `pathsnap/results/route_decoder_comparison.csv`, `route_decoder_summary.json` |
| Official evaluator score (one final run) | `pathsnap/results/evaluator_inference_accuracy.json`, `evaluator_inference_per_trip.csv` |
| Trained models | `pathsnap/models/` |
| Plots and Folium maps | `pathsnap/visuals/` |
All numerical results are generated by the scripts.

## Two-laptop deployment

### Laptop 1 — API and model

For local development using the student-test trips:

```powershell
$env:PATHSNAP_TRIP_PARTITION="student_test"
uvicorn pathsnap.api.main:app --host 0.0.0.0 --port 8000
```

For the evaluator demonstration:

```powershell
$env:PATHSNAP_TRIP_PARTITION="evaluator_test"
uvicorn pathsnap.api.main:app --host 0.0.0.0 --port 8000
```

Allow inbound TCP port 8000 in the firewall and find Laptop 1's LAN IP with `ipconfig`.

### Laptop 2 — UI

```powershell
python -m http.server 8080 --directory pathsnap/ui
```

Open `http://localhost:8080/?api=http://LAPTOP1_LAN_IP:8000`. Both laptops must be on the same network. The API exposes `GET /health`, `GET /trips` and `POST /predict/{trip_id}`. Prediction is completed without reference labels; the endpoint then calculates aggregate Top-1 accuracy and candidate miss rate server-side for the evaluator. It does not return the reference route or per-observation labels.

## Academic integrity and limitations

The route-constrained labels are approximate; the supplied reference route may itself have limitations. GPS gaps, candidate miss rate, disconnected road-network fragments and the local projection approximation affect performance. Do not fabricate results or report evaluator-test metrics from development runs. Team members should understand and be able to explain preprocessing, feature extraction, model comparison, TPE optimisation, emission/transition design and Viterbi decoding.
