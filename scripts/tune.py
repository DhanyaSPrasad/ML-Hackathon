"""Tune the CV-selected classifier with Optuna using training trips only."""
from pathlib import Path
import os,sys,json,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, joblib, optuna
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from pathsnap.modeling import build_model
from pathsnap.core import trip_features, FEATURES
from pathsnap.artifacts import read_model_metadata

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"Student folder"
OUT=ROOT/"pathsnap"/"results"; MOD=ROOT/"pathsnap"/"models"
OUT.mkdir(parents=True,exist_ok=True); MOD.mkdir(parents=True,exist_ok=True)
SEED=42; K=8; POOL=int(os.environ.get("PATHSNAP_CANDIDATE_POOL","64"))
N_TRIALS=int(os.environ.get("PATHSNAP_TUNE_TRIALS","60"))


def partition_path(name):
    for p in (DATA/name,DATA/"train"/name):
        if p.is_dir(): return p
    raise FileNotFoundError(name)


def collect(name):
    xs=[];ys=[];groups=[];trips=[]
    folders=[p for p in sorted(partition_path(name).iterdir()) if p.is_dir() and (p/f"{p.name}.track").is_file()]
    for folder in folders:
        track,nodes,arcs,route,net,X,y,cids,dists,target=trip_features(folder,k=K,candidate_pool=POOL)
        cand=cids.reshape(len(track),K); hit=np.any(cand==target[:,None],axis=1)
        keep=np.repeat(hit,K)
        if keep.any():
            xs.append(X.loc[keep].reset_index(drop=True));ys.append(y[keep])
            groups.extend([folder.name]*int(keep.sum()))
        trips.append((folder.name,X,cids,target,len(track)))
    if not xs: raise RuntimeError("No valid training candidates")
    return pd.concat(xs,ignore_index=True),np.concatenate(ys),np.asarray(groups),trips


def build(trial,kind):
    return build_model(kind, trial=trial)


def score(model,trips,held):
    correct=total=0
    for tid,X,cids,target,n in trips:
        if tid not in held: continue
        p=model.predict_proba(X)[:,1].reshape(n,K)
        cand=cids.reshape(n,K); pred=cand[np.arange(n),np.argmax(p,axis=1)]
        valid=target>=0; correct+=int(np.sum((pred==target)&valid));total+=int(valid.sum())
    return correct/max(total,1)


X,y,groups,train_trips=collect("train")
model_metadata=read_model_metadata(MOD)
selection=json.loads((MOD/"model_selection.json").read_text())
if int(model_metadata["k_candidates"]) != K or int(model_metadata["candidate_pool"]) != POOL:
    raise ValueError("Candidate settings differ from train.py metadata. Re-run train.py or use the same PATHSNAP_CANDIDATE_POOL.")
if selection.get("features") != FEATURES or selection.get("selected_model") != model_metadata.get("selected_classifier"):
    raise ValueError("model_selection.json is stale; rerun scripts/train.py")
kind=selection["selected_model"]
cv=GroupKFold(n_splits=min(5,len(np.unique(groups))))
splits=list(cv.split(X,y,groups))

def objective(trial):
    model=build(trial,kind);fold_scores=[]
    for fit_idx,val_idx in splits:
        model.fit(X.iloc[fit_idx],y[fit_idx])
        fold_scores.append(score(model,train_trips,set(groups[val_idx])))
    return float(np.mean(fold_scores))

study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective,n_trials=N_TRIALS,show_progress_bar=False)
best_trial=optuna.trial.FixedTrial(study.best_params)
model=build_model(kind,params=study.best_params)
start=time.perf_counter();model.fit(X,y);fit_s=time.perf_counter()-start
joblib.dump(model,MOD/"final_model.joblib")
(MOD/"model_metadata.json").write_text(json.dumps({
 "schema_version":2,"features":FEATURES,"k_candidates":K,"candidate_pool":POOL,
 "observations_per_trip":"full resolution","random_seed":SEED,
 "cv":"5-fold GroupKFold grouped by trip ID",
 "selected_classifier":kind,"label_definition":"Monotonic dynamic-programming alignment to ordered reference route; weak geometric supervision.",
 "missed_candidate_policy":"Excluded from classifier fitting; counted as Top-1 errors.",
 "evaluator_partition_used":False},indent=2))

rows=[]
try:
    _,_,_,dev_trips=collect("student_test")
    baseline_path=MOD/(kind.lower().replace(" ","_")+".joblib")
    baseline=joblib.load(baseline_path) if baseline_path.exists() else None
    for label,estimator in [("Untuned",baseline),("Tuned",model)]:
        if estimator is None: continue
        correct=total=0
        for tid,Xt,cids,target,n in dev_trips:
            p=estimator.predict_proba(Xt)[:,1].reshape(n,K)
            cand=cids.reshape(n,K);pred=cand[np.arange(n),np.argmax(p,axis=1)]
            valid=target>=0;correct+=int(np.sum((pred==target)&valid));total+=int(valid.sum())
        rows.append({"model":label,"classifier":kind,"student_test_top1_accuracy":correct/max(total,1)})
except FileNotFoundError:
    pass
pd.DataFrame(rows).to_csv(OUT/"tuned_vs_untuned.csv",index=False)
pd.DataFrame([{"trial":t.number,"value":t.value,**t.params} for t in study.trials]).to_csv(OUT/"bayesian_trials.csv",index=False)
summary={"selected_classifier":kind,"sampler":"Optuna TPE",
 "objective":"5-fold GroupKFold observation-level Top-1 accuracy on training trips",
 "trials":N_TRIALS,"best_params":study.best_params,"best_cv_top1_accuracy":study.best_value,
 "training_trip_count":len(set(groups)),"training_candidate_rows":len(X),
 "candidate_pool":POOL,"k_candidates":K,"features":FEATURES,"schema_version":2,
 "final_fit_seconds":fit_s,"evaluator_partition_used":False}
(OUT/"bayesian_best.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
