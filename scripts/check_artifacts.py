"""Validate that generated model/tuning artifacts match the current pipeline.

This is a local consistency check only. It does not load dataset partitions,
train models, tune parameters, or access evaluator labels.
"""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
from pathsnap.artifacts import read_model_metadata, validate_estimator, read_decoder_metadata

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "pathsnap" / "models"
RESULTS = ROOT / "pathsnap" / "results"


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path.relative_to(ROOT)}. Run the documented pipeline in order."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    metadata = read_model_metadata(MODELS)
    selection = read_json(MODELS / "model_selection.json")
    tuning = read_json(RESULTS / "bayesian_best.json")
    decoder = read_decoder_metadata(MODELS, metadata)
    model_path = MODELS / "final_model.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing {model_path.relative_to(ROOT)}")

    selected = metadata.get("selected_classifier")
    if not selected or selection.get("selected_model") != selected:
        raise ValueError("Selected classifier differs between model metadata and model_selection.json")
    if tuning.get("selected_classifier") != selected:
        raise ValueError("bayesian_best.json does not describe the selected classifier")
    for name, artifact in (("model_selection.json", selection), ("bayesian_best.json", tuning)):
        if artifact.get("features") != metadata["features"]:
            raise ValueError(f"{name} feature schema differs from model_metadata.json")
        if int(artifact.get("k_candidates", 0)) != int(metadata["k_candidates"]):
            raise ValueError(f"{name} candidate count differs from model_metadata.json")
        if int(artifact.get("candidate_pool", 0)) != int(metadata["candidate_pool"]):
            raise ValueError(f"{name} candidate pool differs from model_metadata.json")
        if float(artifact.get("candidate_radius_m", 0)) != float(metadata["candidate_radius_m"]):
            raise ValueError(f"{name} candidate radius differs from model_metadata.json")

    model = joblib.load(model_path)
    validate_estimator(model, metadata)
    print("Artifact check passed.")
    print(f"Classifier: {selected}")
    print(f"Features: {len(metadata['features'])}; candidates: {metadata['k_candidates']}; pool: {metadata['candidate_pool']}")
    print("Model, classifier tuning and decoder metadata are mutually consistent.")


if __name__ == "__main__":
    main()
