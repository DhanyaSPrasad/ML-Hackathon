from pathlib import Path
import sys
import json
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

from pathsnap.core import trip_features, FEATURES


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Student folder"
OUT = ROOT / "pathsnap" / "results"
MOD = ROOT / "pathsnap" / "models"
OUT.mkdir(exist_ok=True, parents=True)
MOD.mkdir(exist_ok=True, parents=True)


def collect(part):
    """Load candidate-level features and labels for every trip in a split."""
    Xs = []
    ys = []
    groups = []
    trips = []

    for folder in sorted((DATA / part).iterdir()):
        if not folder.is_dir():
            continue

        try:
            *_, X, y, candidate_ids, distances, target = trip_features(
                folder, max_observations=120
            )
            if y is None or len(np.unique(y)) < 2:
                continue

            Xs.append(X)
            ys.append(y)
            groups.extend([folder.name] * len(y))
            trips.append(folder.name)

        except Exception as exc:
            print("WARN", folder.name, exc)

    if not Xs:
        raise RuntimeError(
            f"No usable trips were found in {DATA / part}. "
            "Check that the dataset folders are in the expected location."
        )

    return (
        pd.concat(Xs, ignore_index=True),
        np.concatenate(ys),
        np.asarray(groups),
        trips,
    )


print("Loading training trips...")
X, y, g, ids = collect("train")
print("Training data:", X.shape, "Class counts:", np.bincount(y), "Trips:", len(ids))

print("Loading student-test trips...")
Xt, yt, gt, tids = collect("student_test")
print("Student-test data:", Xt.shape, "Class counts:", np.bincount(yt), "Trips:", len(tids))

models = {
    "Logistic Regression": make_pipeline(
        SimpleImputer(),
        StandardScaler(),
        LogisticRegression(max_iter=300, class_weight="balanced", C=1.0),
    ),
    "K-Nearest Neighbours": make_pipeline(
        SimpleImputer(),
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=7, weights="distance"),
    ),
    "Decision Tree": make_pipeline(
        SimpleImputer(),
        DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
        ),
    ),
    "Random Forest": make_pipeline(
        SimpleImputer(),
        RandomForestClassifier(
            n_estimators=40,
            max_depth=16,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
    ),
    "Extra Trees": make_pipeline(
        SimpleImputer(),
        ExtraTreesClassifier(
            n_estimators=40,
            max_depth=18,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
    ),
}

rows = []
cv = GroupKFold(n_splits=3)

for name, model in models.items():
    print("\nFitting", name)
    start_time = time.time()

    try:
        # Cross-validation is performed using training trips only.
        cv_scores = cross_val_score(
            model,
            X,
            y,
            groups=g,
            cv=cv,
            scoring="f1_macro",
            n_jobs=1,
        )

        # Fit the final model on the complete training split.
        model.fit(X, y)

        # Training metrics: predictions on the same data used for fitting.
        train_predictions = model.predict(X)
        train_accuracy = accuracy_score(y, train_predictions)
        train_precision = precision_score(
            y, train_predictions, average="macro", zero_division=0
        )
        train_recall = recall_score(
            y, train_predictions, average="macro", zero_division=0
        )
        train_macro_f1 = f1_score(
            y, train_predictions, average="macro", zero_division=0
        )

        # Student-test metrics: predictions on the held-out student_test split.
        test_predictions = model.predict(Xt)
        test_probabilities = (
            model.predict_proba(Xt) if hasattr(model, "predict_proba") else None
        )

        training_seconds = time.time() - start_time

        row = {
            "model": name,
            "cv_macro_f1": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),

            # Training-set classification metrics (candidate-level).
            "train_accuracy": float(train_accuracy),
            "train_precision": float(train_precision),
            "train_recall": float(train_recall),
            "train_macro_f1": float(train_macro_f1),

            # Held-out student-test metrics.
            "student_top1_accuracy": float(
                accuracy_score(yt, test_predictions)
            ),
            "student_macro_precision": float(
                precision_score(
                    yt, test_predictions, average="macro", zero_division=0
                )
            ),
            "student_macro_recall": float(
                recall_score(
                    yt, test_predictions, average="macro", zero_division=0
                )
            ),
            "student_macro_f1": float(
                f1_score(yt, test_predictions, average="macro", zero_division=0)
            ),

            # Retain the original positive-class precision/recall columns.
            "student_precision": float(
                precision_score(yt, test_predictions, zero_division=0)
            ),
            "student_recall": float(
                recall_score(yt, test_predictions, zero_division=0)
            ),
            "student_log_loss": (
                float(log_loss(yt, test_probabilities, labels=[0, 1]))
                if test_probabilities is not None
                else None
            ),
            "training_seconds": float(training_seconds),
            "inference_seconds": None,
            "config": str(model.get_params()),
        }

        rows.append(row)

        model_filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, MOD / model_filename)
        print(
            f"{name}: train accuracy={train_accuracy:.4f}, "
            f"train macro-F1={train_macro_f1:.4f}, "
            f"student-test Top-1 accuracy={row['student_top1_accuracy']:.4f}, "
            f"student-test macro-F1={row['student_macro_f1']:.4f}"
        )

    except Exception as exc:
        print("MODEL ERROR", name, repr(exc))


results = pd.DataFrame(rows)
results.to_csv(OUT / "model_comparison.csv", index=False)

if not results.empty:
    display_columns = [
        "model",
        "train_accuracy",
        "train_macro_f1",
        "cv_macro_f1",
        "student_top1_accuracy",
        "student_macro_f1",
    ]
    print("\nMODEL COMPARISON")
    print(results[display_columns].to_string(index=False))
else:
    print("No models completed successfully. Check the warnings/errors above.")

metadata = {
    "features": FEATURES,
    "positive_label": "candidate arc is the nearest reference-route arc to the GPS observation",
    "label_alignment": "nearest geometric directed arc among the trip reference-route arcs",
    "k_candidates": 8,
    "training_trip_ids": ids,
    "student_test_trip_ids": tids,
}
(MOD / "model_metadata.json").write_text(json.dumps(metadata, indent=2))

# Bayesian optimisation is run separately with: python scripts/tune.py
