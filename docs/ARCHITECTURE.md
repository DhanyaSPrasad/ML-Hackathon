# PathSnap architecture and data boundaries

```text
Raw GPS track + directed road graph
          |
          v
Full-resolution preprocessing
          |
          v
Local metric projection + KD-tree broad phase
  - radius-complete in-range segment shortlist
  - nearest-midpoint fallback
          |
          v
Exact point-to-segment ranking (50 m default, 8 candidates)
          |
          v
Geometry + motion + previous/next candidate-context features
          |
          v
Ordered reference-route DP alignment (training only)
          |
          v
Five classifiers: Logistic Regression, Decision Tree, Random Forest,
  Extra Trees and LightGBM (KNN removed)
  - 5-fold GroupKFold by trip
  - observation-level Top-1 model selection
          |
          v
Optuna tuning of selected classifier
          |
          v
Classifier scores + movement-aware HMM/Viterbi
          |
          v
Predicted directed road-segment sequence
          |
          v
FastAPI on Laptop 1 <---- HTTP ----> Leaflet UI on Laptop 2
```

## Candidate generation

The KD-tree indexes directed arc midpoints for efficient broad-phase lookup. To avoid dropping long road segments solely because their midpoint is distant, the search radius is expanded by half the longest segment length. All segments in that conservative radius are considered, and a nearest-midpoint shortlist is added as a fallback. Exact point-to-segment distances determine the final candidate order. The model receives the best eight candidates.

## Labels and training

The reference route is an ordered sequence. Training labels use monotonic dynamic-programming alignment between GPS observations and route positions. The cost combines point-to-segment distance and a weak heading term when GPS movement is observable. This remains weak supervision, not independent ground truth. Candidate groups for which the aligned target is absent are excluded from binary classifier fitting; they remain counted as errors in observation-level evaluation.

All temporal features are calculated on the original full-resolution GPS sequence. Downsampling is disabled so training, validation and API inference share the same adjacent-observation semantics.

## Model selection and tuning

Logistic Regression, Decision Tree, Random Forest, Extra Trees and LightGBM are compared with 5-fold GroupKFold, grouping by trip ID. The primary selection metric is observation-level Top-1 accuracy, not flattened candidate-row Macro F1. Optuna TPE tunes the classifier selected using training-only cross-validation. Student-test results are reported separately and are not used to select or tune the model. The evaluator partition is never read by training or tuning.

## HMM / Viterbi

Classifier scores form the emission evidence. The decoder uses directed Dijkstra shortest-path distances between candidate arcs, GPS time gaps, projected position along each segment and observed displacement. Transition scores penalize disagreement between road-network distance and GPS displacement, with a reverse-arc U-turn penalty. Dijkstra results are cached per source node. Gaps over the tuned reset threshold restart the sequence; if all directed transitions are unreachable, the decoder safely restarts at that observation. Decoder parameters should be selected on training-only out-of-fold predictions; continuity is a diagnostic and must not replace Top-1 accuracy as the primary objective.

## Prediction boundary

The API predicts from `.track`, `.nodes` and `.arcs` only. It calls `read_trip(..., include_route=False)` and never supplies reference labels to feature extraction or decoding. If a route file is available, it may be read only after prediction for aggregate evaluation metrics; raw route IDs are not returned to the browser.

## Partition resolution

The code accepts either `Student folder/<partition>` or `Student folder/train/<partition>`. Training-trip discovery requires each folder to contain its own `<trip_id>.track` file, preventing nested partitions from being ingested as training trips.
