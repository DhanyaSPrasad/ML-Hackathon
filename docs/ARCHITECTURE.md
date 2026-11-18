# PathSnap architecture and data boundaries

```text
Raw trip files (.track / .nodes / .arcs / .route)
          |
          v
Data validation and preprocessing
  - numeric coercion / coordinate validation
  - invalid GPS row removal / stable time sorting
  - directed arc endpoint validation
          |
          v
Local metric projection + KD-tree road candidate search
          |
          v
Candidate feature extraction (11 features)
          |
          v
Five classifiers + trip-grouped cross-validation
          |
          v
Random Forest TPE Bayesian optimisation
          |
          v
Candidate scores -> gap-aware HMM/Viterbi decoder
          |
          v
Predicted directed road-segment sequence
          |
          v
FastAPI on Laptop 1 <---- HTTP ----> Leaflet UI on Laptop 2
```

## Offline training/development boundary

The training and student-test scripts may read reference `.route` files to construct weak geometric labels and measure development performance. The positive candidate is the closest segment among the supplied reference-route arcs to a GPS observation. This is an approximation, not independently verified physical ground truth.

## Prediction boundary

The API first calls `read_trip(..., include_route=False)` and `candidate_rows(..., route=None)` to make predictions without reference labels. After route decoding is complete, it may read `.route` to calculate aggregate classifier-only and decoded-route Top-1 accuracy and candidate miss rate. Raw labels do not enter features, model inputs, decoding or the response; only aggregate metrics are returned. For the final demonstration, `PATHSNAP_TRIP_PARTITION=evaluator_test` selects the evaluator trips.

## Partition resolution

The code accepts either `Student folder/<partition>` or `Student folder/train/<partition>`. Training-trip discovery requires each folder to contain its own `<trip_id>.track` file, preventing nested `student_test` or `evaluator_test` folders from being accidentally ingested as training trips.
