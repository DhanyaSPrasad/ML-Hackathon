# PathSnap — Distributed Map-Matching & Route Reconstruction

PathSnap reconstructs directed-road sequences from noisy GPS observations. The repository contains candidate generation, route-aligned weak labels, five required classifiers, Optuna classifier and decoder tuning, HMM/Viterbi decoding, a FastAPI service and a Leaflet UI.

## Dataset layout

```text
Student folder/
├── train/                 # training trips
├── student_test/          # development trips
└── evaluator_test/        # organiser's final evaluation trips
```

The partition resolver supports both `Student folder/<partition>` and `Student folder/train/<partition>`. Each trip folder contains `<trip_id>.track`, `.nodes`, `.arcs` and optionally `.route`.

## Data boundaries

- Training and tuning use only the `train` partition.
- `student_test` is used only for a separate development comparison; it is not used to select or tune models.
- `evaluator_test` is never loaded by training, model selection or tuning.
- Inference first predicts using `.track`, `.nodes` and `.arcs` without loading `.route`. If a reference route is available, the API/final scoring script reads it only after predictions are fixed to calculate aggregate metrics. Raw reference labels are not returned to the UI.

Do not use evaluator reference labels for development, tuning or model selection.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Python 3.10+ is supported. For a quick tuning smoke test, set `$env:PATHSNAP_TUNE_TRIALS="2"` and `$env:PATHSNAP_DECODER_TRIALS="2"`. The full experiment defaults to 60 classifier trials and 60 decoder trials. Training and inference always use the complete original GPS tracks.

## Reproducible workflow

Run from the repository root:

```powershell
python scripts/explore.py
python scripts/train.py
python scripts/tune.py
python scripts/tune_decoder.py
python scripts/evaluate.py
python scripts/visualize.py
python -m unittest discover -s tests -v
```

After model and decoder settings are frozen, run `python scripts/evaluate_final.py` once when the organiser's `evaluator_test` partition is available. Do not use its results to tune or select models.

## Pipeline

1. **Candidate generation** (`pathsnap/core.py`): a KD-tree midpoint index provides a broad-phase shortlist. The search radius is expanded by half the longest segment length, ensuring that every road segment within the configured GPS radius is included. A nearest-midpoint fallback is added, then exact point-to-segment distance ranks candidates. The best eight directed arcs are retained.
2. **Features**: geometric distance, heading difference, road length, GPS speed, time gap, heading change, candidate rank, projection fraction, heading alignment, displacement, circular bearing features and endpoint distances. All features are calculated on full-resolution tracks; training, validation and inference therefore use the same temporal semantics.
3. **Weak labels**: a monotonic dynamic-programming alignment maps observations to the ordered reference-route sequence. It combines point-to-segment distance with weak heading evidence when movement is observable. This is not independently verified ground truth. Observations whose target is absent from their candidate set are excluded from binary classifier fitting, but remain counted as Top-1 errors during evaluation.
4. **Five-model comparison** (`scripts/train.py`): Logistic Regression, KNN, Decision Tree, Random Forest and Extra Trees. Five-fold `GroupKFold` groups by trip ID. The selected model maximizes mean observation-level Top-1 accuracy on training-only folds. Student-test metrics are reported separately.
5. **Classifier tuning** (`scripts/tune.py`): Optuna TPE tunes the selected classifier using training trips only. The tuned model is saved as `pathsnap/models/final_model.joblib`.
6. **Decoder tuning** (`scripts/tune_decoder.py`): creates out-of-fold training predictions and tunes HMM transition penalty, time-gap reset, movement feasibility and emission distance scale. It does not use student-test or evaluator labels.
7. **Evaluation and visualization**: `scripts/evaluate.py` compares classifier-only and HMM/Viterbi accuracy on student-test; `scripts/visualize.py` creates a layered map; `scripts/evaluate_final.py` performs one final evaluator scoring run after the model is frozen.

## Metrics

The primary metric is observation-level Top-1 accuracy: for each GPS observation, select the highest-scoring candidate and compare its directed arc ID with the aligned reference target. Candidate miss rate is an important upper-bound diagnostic: a classifier cannot predict a target absent from the candidate set. Route continuity is secondary; a more connected route is not necessarily a more accurate route.

## Generated outputs

| Output | Location |
|---|---|
| Dataset inventory / network connectivity | `pathsnap/results/dataset_inventory.csv`, `network_connectivity.csv` |
| Five-model comparison | `pathsnap/results/model_comparison.csv` |
| Selected model / feature metadata | `pathsnap/models/model_selection.json`, `model_metadata.json` |
| Classifier tuning history / best parameters | `pathsnap/results/bayesian_trials.csv`, `bayesian_best.json` |
| Decoder tuning history / best parameters | `pathsnap/results/decoder_trials.csv`, `pathsnap/models/decoder_metadata.json` |
| Tuned vs untuned comparison | `pathsnap/results/tuned_vs_untuned.csv` |
| HMM comparison | `pathsnap/results/route_decoder_comparison.csv`, `route_decoder_summary.json` |
| Evaluator score (one final run) | `pathsnap/results/evaluator_inference_accuracy.json`, `evaluator_inference_per_trip.csv` |
| Models | `pathsnap/models/` |
| Folium map | `pathsnap/visuals/` |

## Two-laptop deployment

### Laptop 1 — API

For development trips:

```powershell
$env:PATHSNAP_TRIP_PARTITION="student_test"
uvicorn pathsnap.api.main:app --host 0.0.0.0 --port 8000
```

For the final evaluator demonstration:

```powershell
$env:PATHSNAP_TRIP_PARTITION="evaluator_test"
uvicorn pathsnap.api.main:app --host 0.0.0.0 --port 8000
```

Allow inbound TCP port 8000 in the firewall and find Laptop 1's LAN IP with `ipconfig`.

### Laptop 2 — UI

```powershell
python -m http.server 8080 --directory pathsnap/ui
```

Open `http://localhost:8080/?api=http://LAPTOP1_LAN_IP:8000`. Both laptops must be on the same network. The API exposes `GET /health`, `GET /trips` and `POST /predict/{trip_id}`.

## Limitations

Reference-route alignment is weak supervision, not independent ground truth. GPS noise, missing candidate roads, network fragmentation and the local projection approximation can affect performance. Do not claim an evaluator accuracy until the organiser's evaluator partition has actually been scored. Team members should be able to explain candidate generation, route alignment, all five classifiers, Optuna, HMM emissions/transitions and Viterbi decoding.
