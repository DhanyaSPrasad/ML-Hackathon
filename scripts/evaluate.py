"""Compare classifier-only and HMM/Viterbi decoding per model on student_test."""
from pathlib import Path
import os
import sys
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from pathsnap.core import trip_features, viterbi

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Student folder"
OUT = ROOT / "pathsnap" / "results"
MODEL_DIR = ROOT / "pathsnap" / "models"
K = 8


def partition_path(name):
    for path in (DATA / name, DATA / "train" / name):
        if path.is_dir():
            return path
    raise FileNotFoundError(name)


def continuity(sequence, arcs):
    edges = arcs[["from_node", "to_node"]].to_numpy(dtype=int)
    good = total = 0
    for previous, current in zip(sequence[:-1], sequence[1:]):
        if 0 <= previous < len(edges) and 0 <= current < len(edges):
            total += 1
            good += int(edges[previous, 1] == edges[current, 0])
    return good / max(total, 1)


folders = [p for p in sorted(partition_path("student_test").iterdir())
           if p.is_dir() and (p / f"{p.name}.track").is_file()]
models = []
for path in sorted(MODEL_DIR.glob("*.joblib")):
    k_model = K
    models.append((path.stem.replace("_", " ").title(), joblib.load(path), k_model))
if not folders:
    raise RuntimeError("No student_test trips found")
if not models:
    raise RuntimeError(f"No saved models found in {MODEL_DIR}")

rows = []
for model_name, model, k_model in models:
    for folder in folders:
        track, nodes, arcs, route, net, X, y, cids, dists, target = trip_features(
            folder, k=k_model,
            max_observations=int(os.environ.get("PATHSNAP_OBS_PER_TRIP", "120")))
        probabilities = model.predict_proba(X)[:, 1]
        scores = probabilities.reshape(len(track), k_model)
        scores = scores / (scores.sum(axis=1, keepdims=True) + 1e-9)
        candidates = cids.reshape(len(track), k_model)
        classifier = candidates[np.arange(len(track)), np.argmax(scores, axis=1)]
        gaps = np.r_[0.0, np.diff(track.timestamp.to_numpy(dtype=float))]
        decoded = viterbi(cids, dists, scores.ravel(), arcs, k=k_model,
                          time_gaps=gaps, gap_reset_s=5.0)
        miss = np.asarray([target[i] not in candidates[i] for i in range(len(track))])
        rows.append({
            "model": model_name,
            "trip_id": folder.name,
            "observations": len(track),
            "candidate_miss_rate": float(miss.mean()),
            "classifier_top1_accuracy": float(accuracy_score(target, classifier)),
            "hmm_accuracy": float(accuracy_score(target, decoded)),
            "classifier_route_continuity": continuity(classifier, arcs),
            "hmm_route_continuity": continuity(decoded, arcs),
            "hmm_sequence_exact_match": float(np.array_equal(target, decoded)),
        })

df = pd.DataFrame(rows)
df.to_csv(OUT / "route_decoder_comparison.csv", index=False)
summary = df.groupby("model").mean(numeric_only=True).to_dict(orient="index")
(OUT / "route_decoder_summary.json").write_text(json.dumps(summary, indent=2))
columns = ["classifier_top1_accuracy", "hmm_accuracy",
           "classifier_route_continuity", "hmm_route_continuity"]
print(df.groupby("model")[columns].mean().to_string())
