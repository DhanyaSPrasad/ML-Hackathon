"""Optuna TPE optimisation of Random Forest using training trips only."""
from pathlib import Path
import os
import sys, json, time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from pathsnap.core import trip_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Student folder"; OUT = ROOT / "pathsnap" / "results"; MOD = ROOT / "pathsnap" / "models"
OUT.mkdir(parents=True, exist_ok=True); MOD.mkdir(parents=True, exist_ok=True)
SEED = 42; K = 8; OBS_PER_TRIP = int(os.environ.get("PATHSNAP_OBS_PER_TRIP", "120")); N_TRIALS = int(os.environ.get("PATHSNAP_TUNE_TRIALS", "30"))

def partition_path(name):
    for p in (DATA / name, DATA / "train" / name):
        if p.is_dir(): return p
    raise FileNotFoundError(f"Missing {name} partition")

def collect(name):
    xs, ys, groups, trip_data = [], [], [], []
    base = partition_path(name)
    folders = [p for p in sorted(base.iterdir()) if p.is_dir() and (p / f"{p.name}.track").is_file()]
    for folder in folders:
        track, nodes, arcs, route, net, X, y, cids, dists, target = trip_features(folder, k=K, max_observations=OBS_PER_TRIP)
        if y is None or len(np.unique(y)) < 2: continue
        xs.append(X); ys.append(y); groups.extend([folder.name] * len(y))
        trip_data.append((folder.name, arcs, X, cids, target, len(track)))
    return pd.concat(xs, ignore_index=True), np.concatenate(ys), np.asarray(groups), trip_data

def build_model(params):
    return make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(
        **params, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED))

print("Loading 80 training trips for Bayesian search...", flush=True)
X, y, groups, train_trips = collect("train")
print("Loading 10 student-test trips for development evaluation...", flush=True)
Xt, yt, gt, test_trips = collect("student_test")
cv = GroupKFold(n_splits=3)

def cv_top1_score(model):
    scores = []
    for fit_idx, valid_idx in cv.split(X, y, groups):
        model.fit(X.iloc[fit_idx], y[fit_idx])
        valid_groups = set(groups[valid_idx])
        correct = total = 0
        for trip_id, arcs, X_trip, cids, target, n in train_trips:
            if trip_id not in valid_groups:
                continue
            probabilities = model.predict_proba(X_trip)[:, 1].reshape(n, K)
            candidates = cids.reshape(n, K)
            chosen = candidates[np.arange(n), np.argmax(probabilities, axis=1)]
            correct += int(np.sum(chosen == target))
            total += n
        scores.append(correct / max(total, 1))
    return float(np.mean(scores))


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 60, 180),
        "max_depth": trial.suggest_int("max_depth", 8, 24),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.7, 0.9]),
    }
    model = build_model(params)
    return cv_top1_score(model)

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

# Fit final tuned model on the full permitted training partition.
model = build_model(study.best_params)
fit_start = time.perf_counter(); model.fit(X, y); fit_seconds = time.perf_counter() - fit_start
joblib.dump(model, MOD / "final_model.joblib")

# Compare tuned and untuned RF on exactly the same student-test candidate rows.
untuned = build_model({"n_estimators":100,"max_depth":16,"min_samples_leaf":3})
untuned_fit_start = time.perf_counter()
untuned.fit(X, y)
untuned_fit_seconds = time.perf_counter() - untuned_fit_start

# Use the same trip-grouped Top-1 objective for baseline and tuned RF.
untuned_cv = cv_top1_score(build_model({"n_estimators":100,"max_depth":16,"min_samples_leaf":3}))

def evaluate(name, estimator):
    t0 = time.perf_counter(); pred = estimator.predict(Xt); proba = estimator.predict_proba(Xt)[:, 1]; infer = time.perf_counter()-t0
    top_correct = total = cont_good = cont_total = 0
    for trip_id, arcs, X_trip, cids, target, n in test_trips:
        p = estimator.predict_proba(X_trip)[:, 1].reshape(n, K)
        candidates = cids.reshape(n, K)
        chosen = candidates[np.arange(n), np.argmax(p, axis=1)]
        top_correct += int(np.sum(chosen == target)); total += n
        edges = arcs[["from_node","to_node"]].to_numpy(dtype=int)
        for a,b in zip(chosen[:-1],chosen[1:]):
            if 0 <= a < len(edges) and 0 <= b < len(edges):
                cont_total += 1; cont_good += int(edges[a,1] == edges[b,0])
    return {
        "model": name,
        "cv_top1_accuracy": study.best_value if name == "Tuned Random Forest" else untuned_cv,
        "candidate_accuracy": float(accuracy_score(yt,pred)),
        "precision_macro": float(precision_score(yt,pred,average="macro",zero_division=0)),
        "recall_macro": float(recall_score(yt,pred,average="macro",zero_division=0)),
        "candidate_macro_f1": float(f1_score(yt,pred,average="macro",zero_division=0)),
        "log_loss": float(log_loss(yt,proba,labels=[0,1])),
        "observation_top1_accuracy": top_correct/max(total,1),
        "route_continuity": cont_good/max(cont_total,1),
        "inference_seconds": infer,
        "training_seconds": fit_seconds if name == "Tuned Random Forest" else untuned_fit_seconds,
    }

comparison = pd.DataFrame([evaluate("Untuned Random Forest",untuned),evaluate("Tuned Random Forest",model)])
comparison.to_csv(OUT / "tuned_vs_untuned.csv", index=False)
trials = pd.DataFrame([{"trial": t.number, "cv_top1_accuracy": t.value, **t.params} for t in study.trials])
trials.to_csv(OUT / "bayesian_trials.csv", index=False)
summary = {
    "sampler":"Optuna TPESampler", "objective":"3-fold GroupKFold observation-level Top-1 accuracy, grouped by trip ID",
    "trials":N_TRIALS,"best_params":study.best_params,"best_cv_top1_accuracy":study.best_value,
    "training_trip_count":len(np.unique(groups)),"training_candidate_rows":len(X),
    "observations_per_training_trip":OBS_PER_TRIP,"final_fit_seconds":fit_seconds,
    "evaluator_partition_used":False,
}
(OUT / "bayesian_best.json").write_text(json.dumps(summary,indent=2))
print("Best parameters:", study.best_params)
print("Best trip-grouped CV Top-1 accuracy:", study.best_value)
print(comparison.to_string(index=False))
