"""One-shot evaluator_test scoring after the model/configuration are frozen.

Inference is completed with reference routes excluded. Reference labels are
read afterward only to calculate and save aggregate accuracy metrics.
Never use this output to tune or select a model.
"""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from pathsnap.core import read_trip, make_network, candidate_rows, viterbi
from pathsnap.artifacts import read_model_metadata, validate_estimator

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Student folder"
MODELS = ROOT / "pathsnap" / "models"
OUT = ROOT / "pathsnap" / "results"
metadata = read_model_metadata(MODELS)
MODEL = validate_estimator(joblib.load(MODELS / "final_model.joblib"), metadata)
k = int(metadata.get("k_candidates", 8))
decoder = json.loads((MODELS / "decoder_metadata.json").read_text()).get("best_params", {})


def partition_path():
    for path in (DATA / "evaluator_test", DATA / "train" / "evaluator_test"):
        if path.is_dir():
            return path
    raise FileNotFoundError("evaluator_test partition was not found")


def infer_without_labels(folder):
    track, nodes, arcs, _ = read_trip(folder, include_route=False)
    network = make_network(nodes, arcs)
    X, _, candidate_ids, distances, _ = candidate_rows(
        track, network, route=None, k=k,
        candidate_pool=int(metadata.get("candidate_pool", 64)))
    probabilities = MODEL.predict_proba(X)[:, 1]
    scores = probabilities.reshape(len(track), k)
    scores = scores / (scores.sum(axis=1, keepdims=True) + 1e-9)
    candidates = candidate_ids.reshape(len(track), k)
    classifier = candidates[np.arange(len(track)), np.argmax(scores, axis=1)]
    gaps = np.r_[0.0, np.diff(track.timestamp.to_numpy(dtype=float))]
    arc_lengths = np.zeros(len(arcs), dtype=float)
    arc_lengths[network["ids"]] = network["lengths"]
    fractions = X["projection_fraction"].to_numpy(dtype=float)
    displacement = X["gps_displacement_m"].to_numpy(dtype=float).reshape(len(track), k)[:, 0]
    decoded = viterbi(
        candidate_ids, distances, scores.reshape(-1), arcs, k=k,
        time_gaps=gaps, transition_penalty=float(decoder.get("transition_penalty", 3.0)),
        gap_reset_s=float(decoder.get("gap_reset_s", 5.0)),
        projection_fractions=fractions, arc_lengths=arc_lengths,
        gps_displacements=displacement,
        speed_weight=float(decoder.get("speed_weight", 0.015)),
        max_speed_mps=float(decoder.get("max_speed_mps", 55.0)),
        distance_scale_m=float(decoder.get("distance_scale_m", 80.0)),
    )
    return track, network, candidates, classifier, decoded


base = partition_path()
folders = [p for p in sorted(base.iterdir())
           if p.is_dir() and (p / f"{p.name}.track").is_file()]
if not folders:
    raise RuntimeError("evaluator_test contains no trip folders")

rows = []
total_valid = total_miss = classifier_correct = decoded_correct = 0
for folder in folders:
    # First produce predictions with no reference route available.
    track, network, candidates, classifier, decoded = infer_without_labels(folder)

    # Only after predictions are fixed, read the reference route for scoring.
    _, _, _, reference_route = read_trip(folder, include_route=True)
    _, _, _, _, target = candidate_rows(
        track, network, route=reference_route, k=k,
        candidate_pool=int(metadata.get("candidate_pool", 64)))
    valid = target >= 0
    miss = np.asarray([target[i] not in candidates[i] for i in range(len(track))])
    n_valid = int(valid.sum())
    n_miss = int(miss[valid].sum())
    n_classifier = int(np.sum(classifier[valid] == target[valid]))
    n_decoded = int(np.sum(decoded[valid] == target[valid]))
    total_valid += n_valid
    total_miss += n_miss
    classifier_correct += n_classifier
    decoded_correct += n_decoded
    rows.append({
        "trip_id": folder.name,
        "observations": n_valid,
        "candidate_miss_rate": n_miss / max(n_valid, 1),
        "classifier_top1_accuracy": n_classifier / max(n_valid, 1),
        "decoded_top1_accuracy": n_decoded / max(n_valid, 1),
    })

per_trip = pd.DataFrame(rows)
per_trip.to_csv(OUT / "evaluator_inference_per_trip.csv", index=False)
summary = {
    "partition": "evaluator_test",
    "model_artifact": "pathsnap/models/final_model.joblib",
    "k_candidates": k,
    "decoder": decoder,
    "observations": total_valid,
    "candidate_miss_rate": total_miss / max(total_valid, 1),
    "classifier_top1_accuracy": classifier_correct / max(total_valid, 1),
    "decoded_top1_accuracy": decoded_correct / max(total_valid, 1),
    "reference_labels_used_for_inference_or_selection": False,
}
(OUT / "evaluator_inference_accuracy.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
