# PathSnap — Distributed Map-Matching & Route Reconstruction

A compact end-to-end implementation for the PES University UE24CS352A ML hackathon. It loads per-trip GPS tracks and directed road networks, generates nearby road-segment candidates, trains and compares five classifiers, tunes a Random Forest with Optuna's TPE Bayesian sampler, decodes candidate sequences with Viterbi, and serves inference through FastAPI with a separate browser UI.

## Dataset layout

The supplied data is included under `Student folder/`:

- `train/`: 80 trips used for fitting and group-aware cross-validation.
- `student_test/`: 10 trips used only for development-stage comparisons and the demonstration API.
- Each trip has `.track`, `.nodes`, `.arcs`, and `.route` files.

**Important:** The uploaded dataset does not contain the separate evaluator-testing partition described in the hackathon guide. The UI currently lists the ten supplied student-test trips for development demonstration. For the final event, place the organiser-provided evaluator trips in `Student folder/evaluator_test/` and change the API partition setting to that directory. Do not use evaluator routes in training or tuning.

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.

## Run the pipeline

From the project root:

```bash
python scripts/explore.py
python scripts/train.py
python scripts/refresh_model_metrics.py
python scripts/tune.py
python scripts/evaluate.py
python scripts/visualize.py
python scripts/build_report.py
```

The scripts create the dataset inventory and connectivity summaries, train and compare five models, correct the observation-level Top-1 metric, run a compact Bayesian TPE search, evaluate classifier-only versus HMM decoding, generate plots/Folium maps, and build the technical report PDF. Initial five-model training is the longest step and can take several minutes on CPU. All metrics are generated from the actual supplied files; do not replace them with illustrative values. `refresh_model_metrics.py` is required after `train.py` because it computes the required observation-level Top-1 metric from the candidate groups.

## Start API and UI (two laptops)

**Laptop 1 — model and API:**

```bash
uvicorn pathsnap.api.main:app --host 0.0.0.0 --port 8000
```

Allow inbound TCP port 8000 in the laptop firewall. Find Laptop 1's LAN IP using `ipconfig` (Windows) or `hostname -I` (Linux).

**Laptop 2 — UI:** Copy/clone the project (the UI folder is sufficient) and serve it:

```bash
python -m http.server 8080 --directory pathsnap/ui
```

Open `http://localhost:8080/?api=http://LAPTOP1_LAN_IP:8000` in a browser, replacing the address with Laptop 1's LAN IP. Both laptops should be on the same network. The API logs each inference request and completion.

API endpoints: `GET /health`, `GET /trips`, `POST /predict/{trip_id}`.

## Method overview and assumptions

1. Each `.track` row is interpreted as longitude, latitude, timestamp; `.nodes` as longitude/latitude; `.arcs` as directed from-node/to-node rows; `.route` values are zero-based arc-row IDs.
2. Candidate generation queries the eight nearest directed-arc midpoints in a local metric approximation. Features include GPS-to-midpoint distance, heading difference, road length/bearing, speed, time gap and candidate rank.
3. Because the reference route is an ordered arc sequence and the supplied files do not provide a direct timestamp-to-route-arc alignment, training labels use a documented linear-progress alignment from GPS observation index to route sequence index. This is a weak-label approximation and a limitation; the report must discuss it. Candidate miss rate should be interpreted accordingly.
4. Five classifiers are compared: Logistic Regression, KNN, Decision Tree, Random Forest and Extra Trees. GroupKFold keeps trips together.
5. Optuna TPE optimises Random Forest macro-F1 over group-aware folds. The trial history is retained.
6. Viterbi uses classifier candidate scores as emissions and a directed-edge connectivity reward/penalty as transitions. Disconnected transitions are penalised rather than silently treated as valid.
7. The API reads reference routes internally for aggregate development evaluation. The browser receives predictions and aggregate metrics only; it never receives reference route IDs.

## Deliverables

- Source: `pathsnap/`, `scripts/`
- Model artifacts: `pathsnap/models/`
- Experiment logs/tables: `pathsnap/results/`
- Visualisations: `pathsnap/visuals/`
- Technical report: `pathsnap/report/`
- This README and `requirements.txt`

Run all scripts yourself before submission and review the generated results. The academic-integrity rules require that every team member understands and can defend the implementation and actual results.

## Report and result files

The generated report is `pathsnap/report/PathSnap_Technical_Report.pdf`. The experiment tables and logs are under `pathsnap/results/`; plots and the Folium map are under `pathsnap/visuals/`. Review the PDF and all result tables before submission.

## Current measured development snapshot

On the supplied student-test partition, the initial Random Forest achieved approximately 0.865 group-CV candidate Macro F1, 0.785 observation-level Top-1 Accuracy, and 0.870 student-test candidate Macro F1. The tuned model's candidate Macro F1 was approximately 0.864 and its observation-level Top-1 Accuracy approximately 0.773. Across the 10 development trips, the HMM improved directed route continuity (about 0.248 to 0.458) while reducing pointwise route accuracy (about 0.773 to 0.703). These are development results under the documented route-constrained nearest-segment labels, not final evaluator results.
