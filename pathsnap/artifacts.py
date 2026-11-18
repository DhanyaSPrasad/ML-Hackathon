"""Shared validation for model artifacts and the current feature schema."""
import json
from pathlib import Path

from pathsnap.core import FEATURES

SCHEMA_VERSION = 3


def read_model_metadata(models_dir):
    path = Path(models_dir) / "model_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/train.py and scripts/tune.py first."
        )
    metadata = json.loads(path.read_text(encoding="utf-8"))
    validate_metadata(metadata)
    return metadata


def validate_metadata(metadata):
    expected = list(FEATURES)
    actual = metadata.get("features")
    if actual != expected:
        raise ValueError(
            "Model feature schema is stale or reordered. "
            f"Expected {expected!r}; found {actual!r}. "
            "Regenerate artifacts with scripts/train.py, then scripts/tune.py."
        )
    if metadata.get("observations_per_trip") != "full resolution":
        raise ValueError(
            "Model metadata does not declare full-resolution observations. "
            "Regenerate artifacts with the current training pipeline."
        )
    k = int(metadata.get("k_candidates", 0))
    pool = int(metadata.get("candidate_pool", 0))
    radius = float(metadata.get("candidate_radius_m", 0))
    if k < 1 or pool < k or radius <= 0:
        raise ValueError("Invalid candidate count, pool or radius in model_metadata.json")
    return metadata


def validate_estimator(model, metadata):
    validate_metadata(metadata)
    expected_count = len(FEATURES)
    actual_count = getattr(model, "n_features_in_", None)
    if actual_count is not None and int(actual_count) != expected_count:
        raise ValueError(
            f"Model expects {actual_count} features but current pipeline emits "
            f"{expected_count}. Retrain and retune the model."
        )
    actual_names = getattr(model, "feature_names_in_", None)
    if actual_names is not None and list(actual_names) != list(FEATURES):
        raise ValueError(
            "Serialized model feature names/order do not match pathsnap.core.FEATURES."
        )
    return model


def read_decoder_metadata(models_dir, model_metadata):
    """Load decoder settings only when they were tuned for this exact model setup."""
    path = Path(models_dir) / "decoder_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run scripts/tune_decoder.py first.")
    decoder = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "classifier": model_metadata.get("selected_classifier"),
        "features": list(FEATURES),
        "k_candidates": int(model_metadata["k_candidates"]),
        "candidate_pool": int(model_metadata["candidate_pool"]),
        "candidate_radius_m": float(model_metadata["candidate_radius_m"]),
    }
    for key, value in expected.items():
        if decoder.get(key) != value:
            raise ValueError(
                f"decoder_metadata.json {key}={decoder.get(key)!r} does not match "
                f"the current model configuration {value!r}. Rerun scripts/tune_decoder.py."
            )
    return decoder
