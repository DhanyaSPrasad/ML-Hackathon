"""Train and compare five classifiers using trip-grouped validation."""
from pathlib import Path
import os
import sys, json, time, warnings
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             log_loss, confusion_matrix, ConfusionMatrixDisplay)
from pathsnap.core import trip_features, FEATURES

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Student folder"
OUT = ROOT / "pathsnap" / "results"
MOD = ROOT / "pathsnap" / "models"
VIS = ROOT / "pathsnap" / "visuals"
for path in (OUT, MOD, VIS): path.mkdir(parents=True, exist_ok=True)
SEED = 42
OBS_PER_TRIP = int(os.environ.get("PATHSNAP_OBS_PER_TRIP", "120"))
K = 8


def partition_path(name):
    for path in (DATA / name, DATA / "train" / name):
        if path.is_dir(): return path
    raise FileNotFoundError(f"Missing partition {name!r}; checked {DATA/name} and {DATA/'train'/name}")


def collect(partition):
    Xs, ys, groups, trip_data, ids = [], [], [], [], []
    base = partition_path(partition)
    folders = [p for p in sorted(base.iterdir()) if p.is_dir() and (p / f"{p.name}.track").is_file()]
    for folder in folders:
        try:
            track, nodes, arcs, route, net, X, y, cids, dists, target = trip_features(folder, k=K, max_observations=OBS_PER_TRIP)
            if y is None or len(np.unique(y)) < 2: continue
            Xs.append(X); ys.append(y); groups.extend([folder.name] * len(y)); ids.append(folder.name)
            trip_data.append((folder.name, arcs, X, cids, y, target, len(track)))
        except Exception as exc:
            print("WARN", partition, folder.name, repr(exc))
    if not Xs: raise RuntimeError(f"No usable trips in {base}")
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys), np.asarray(groups), ids, trip_data


def metric_pack(y_true, y_pred, probabilities=None):
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_positive": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_positive": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    if probabilities is not None:
        result["log_loss"] = float(log_loss(y_true, probabilities, labels=[0, 1]))
    return result


def observation_top1(model, trip_data):
    correct = total = 0
    continuity_good = continuity_total = 0
    for trip_id, arcs, X_trip, cids, y_trip, target, n_obs in trip_data:
        probs = model.predict_proba(X_trip)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_trip).astype(float)
        cand = cids.reshape(n_obs, K)
        chosen = cand[np.arange(n_obs), np.argmax(probs.reshape(n_obs, K), axis=1)]
        correct += int(np.sum(chosen == target)); total += len(target)
        edge = arcs[["from_node", "to_node"]].to_numpy(dtype=int)
        for a, b in zip(chosen[:-1], chosen[1:]):
            if 0 <= a < len(edge) and 0 <= b < len(edge):
                continuity_total += 1
                continuity_good += int(edge[a, 1] == edge[b, 0])
    return correct / max(total, 1), continuity_good / max(continuity_total, 1)

print("Loading training trips (80-trip partition only)...", flush=True)
X, y, groups, train_ids, _ = collect("train")
print("Training matrix:", X.shape, "trips:", len(train_ids), "class counts:", np.bincount(y), flush=True)
print("Loading student-test trips for development comparison...", flush=True)
Xt, yt, gt, test_ids, test_trip_data = collect("student_test")
print("Student-test matrix:", Xt.shape, "trips:", len(test_ids), flush=True)

models = {
    "Logistic Regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced", C=1.0, random_state=SEED)),
    "K-Nearest Neighbours": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), KNeighborsClassifier(n_neighbors=7, weights="distance")),
    "Decision Tree": make_pipeline(SimpleImputer(strategy="median"), DecisionTreeClassifier(max_depth=12, min_samples_leaf=4, class_weight="balanced", random_state=SEED)),
    "Random Forest": make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)),
    "Extra Trees": make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(n_estimators=100, max_depth=18, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=SEED)),
}
cv = GroupKFold(n_splits=3)
rows = []
for name, model in models.items():
    print("\nFitting", name, flush=True)
    cv_start = time.perf_counter()
    cv_scores = cross_val_score(model, X, y, groups=groups, cv=cv, scoring="f1_macro", n_jobs=1)
    cv_seconds = time.perf_counter() - cv_start
    fit_start = time.perf_counter()
    model.fit(X, y)
    fit_seconds = time.perf_counter() - fit_start
    train_pred = model.predict(X)
    train_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    test_start = time.perf_counter()
    test_pred = model.predict(Xt)
    test_prob = model.predict_proba(Xt)[:, 1] if hasattr(model, "predict_proba") else None
    inference_seconds = time.perf_counter() - test_start
    train_metrics = metric_pack(y, train_pred, train_prob)
    test_metrics = metric_pack(yt, test_pred, test_prob)
    # Top-1 and continuity are computed per observation, not over flattened candidates.
    top1, continuity = observation_top1(model, test_trip_data)
    row = {"model": name, "algorithm_family": {
        "Logistic Regression":"linear/probabilistic", "K-Nearest Neighbours":"instance-based",
        "Decision Tree":"tree", "Random Forest":"ensemble", "Extra Trees":"ensemble"}[name],
        "cv_macro_f1_mean": float(cv_scores.mean()), "cv_macro_f1_std": float(cv_scores.std()),
        "train_accuracy": train_metrics["accuracy"], "train_precision_macro": train_metrics["precision_macro"],
        "train_recall_macro": train_metrics["recall_macro"], "train_macro_f1": train_metrics["f1_macro"],
        # Candidate-row accuracy and observation-level route accuracy are
        # different metrics; publish explicit test names for both.
        "student_test_candidate_accuracy": test_metrics["accuracy"],
        "student_candidate_accuracy": test_metrics["accuracy"], "student_precision_macro": test_metrics["precision_macro"],
        "student_recall_macro": test_metrics["recall_macro"], "student_macro_f1": test_metrics["f1_macro"],
        "student_precision_positive": test_metrics["precision_positive"], "student_recall_positive": test_metrics["recall_positive"],
        "student_log_loss": test_metrics.get("log_loss"), "student_top1_accuracy": top1,
        "student_route_continuity": continuity, "cv_seconds": cv_seconds,
        "training_seconds": fit_seconds, "inference_seconds_total": inference_seconds, "inference_ms_per_candidate": 1000*inference_seconds/max(len(Xt),1),
        "config": json.dumps(model.get_params(), default=str),
    }
    rows.append(row)
    filename = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(model, MOD / filename)
    cm = confusion_matrix(yt, test_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Other candidate", "Reference-route candidate"])
    fig, ax = plt.subplots(figsize=(5, 4)); disp.plot(ax=ax, colorbar=False, values_format="d")
    ax.set_title(f"{name} — student-test candidate confusion matrix"); fig.tight_layout()
    fig.savefig(VIS / f"confusion_{filename.replace('.joblib','.png')}", dpi=160); plt.close(fig)
    print(f"{name}: train candidate acc={train_metrics['accuracy']:.4f}, student-test candidate acc={test_metrics['accuracy']:.4f}, student-test Top-1 accuracy={top1:.4f}, CV Macro F1={cv_scores.mean():.4f}", flush=True)

results = pd.DataFrame(rows)
results.to_csv(OUT / "model_comparison.csv", index=False)
# Record feature names and preprocessing/model configuration with the selected baseline.
(MOD / "model_metadata.json").write_text(json.dumps({
    "features": FEATURES, "k_candidates": K, "observations_per_trip": OBS_PER_TRIP,
    "random_seed": SEED, "cv": "3-fold GroupKFold grouped by trip ID",
    "label_definition": "For each GPS observation, the closest road segment among the supplied reference-route arcs is the positive candidate (weak geometric label).",
    "warning": "Student-test metrics are development-stage metrics; evaluator_test is never loaded by this script."
}, indent=2))

for metric, title, filename in [
    ("cv_macro_f1_mean", "3-fold trip-grouped CV Macro F1", "cv_macro_f1.png"),
    ("student_top1_accuracy", "Student-test observation-level Top-1 Accuracy", "student_accuracy.png"),
    ("student_macro_f1", "Student-test candidate-level Macro F1", "student_macro_f1.png"),
    ("student_log_loss", "Student-test candidate-level Log Loss", "student_log_loss.png"),
]:
    fig, ax = plt.subplots(figsize=(9, 4.8)); ax.bar(results.model, results[metric]); ax.set_title(title); ax.set_ylabel(metric); ax.tick_params(axis="x", rotation=20); fig.tight_layout(); fig.savefig(VIS / filename, dpi=160); plt.close(fig)

# Tree feature importance is a diagnostic only; selection is not performed on student-test labels.
for name in ("Random Forest", "Extra Trees"):
    estimator = models[name].steps[-1][1]
    if hasattr(estimator, "feature_importances_"):
        importance = pd.DataFrame({"feature": FEATURES, "importance": estimator.feature_importances_}).sort_values("importance", ascending=False)
        importance.to_csv(OUT / f"feature_importance_{name.lower().replace(' ', '_')}.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 5)); ax.barh(importance.feature[::-1], importance.importance[::-1]); ax.set_title(f"{name} feature importance"); fig.tight_layout(); fig.savefig(VIS / f"feature_importance_{name.lower().replace(' ', '_')}.png", dpi=160); plt.close(fig)

print("\nMODEL COMPARISON")
print(results[["model", "train_accuracy", "student_test_candidate_accuracy", "student_top1_accuracy", "cv_macro_f1_mean", "student_macro_f1", "student_route_continuity"]].to_string(index=False))
print("Results:", OUT / "model_comparison.csv")
