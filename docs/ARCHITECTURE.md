# PathSnap architecture

```text
Per-trip .track/.nodes/.arcs/.route files
                  |
                  v
       Data loader and EDA scripts
                  |
                  v
       KD-tree candidate generation
                  |
                  v
       Candidate feature extraction
                  |
                  v
     Five classifier experiments
                  |
                  v
    Random Forest + Bayesian TPE tune
                  |
                  v
       Candidate probability scores
                  |
                  v
        HMM / Viterbi decoder
                  |
                  v
       Predicted directed arc route
                  |
                  v
      FastAPI (Laptop 1, port 8000)
                  ^
                  | HTTP request/response
                  v
      Leaflet UI (Laptop 2, port 8080)
```

Reference `.route` values are used only in offline training/development labels and server-side aggregate evaluation. The UI receives predicted arc IDs/geometry and aggregate metrics, not reference route IDs. The supplied archive has no separate evaluator partition; the API partition is configurable through `PATHSNAP_TRIP_PARTITION`.
