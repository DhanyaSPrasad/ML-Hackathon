"""Optuna tuning for the selected classifier using training trips only."""
from pathlib import Path
import os,sys,json,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,joblib,optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from pathsnap.core import trip_features,FEATURES
from pathsnap.artifacts import read_model_metadata

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/"Student folder"
OUT=ROOT/"pathsnap"/"results";MOD=ROOT/"pathsnap"/"models"
OUT.mkdir(parents=True,exist_ok=True);MOD.mkdir(parents=True,exist_ok=True)
SEED=42;TRIALS=int(os.environ.get("PATHSNAP_TUNE_TRIALS","50"))


def partition_path(name):
    for p in (DATA/name,DATA/"train"/name):
        if p.is_dir():return p
    raise FileNotFoundError(name)


def collect(name,meta):
    k=int(meta["k_candidates"]);pool=int(meta["candidate_pool"]);radius=float(meta["candidate_radius_m"])
    xs=[];ys=[];groups=[];trips=[]
    folders=[p for p in sorted(partition_path(name).iterdir()) if p.is_dir() and (p/f"{p.name}.track").is_file()]
    for folder in folders:
        track,nodes,arcs,route,net,X,y,cids,dists,target=trip_features(folder,k=k,candidate_pool=pool,radius_m=radius)
        cand=cids.reshape(len(track),k);hit=np.any(cand==target[:,None],axis=1);keep=np.repeat(hit,k)
        if keep.any():
            xs.append(X.loc[keep].reset_index(drop=True));ys.append(y[keep]);groups.extend([folder.name]*int(keep.sum()))
        trips.append({"id":folder.name,"X":X,"cids":cids,"target":target,"n":len(track)})
    if not xs:raise RuntimeError("No valid training candidate rows")
    return pd.concat(xs,ignore_index=True),np.concatenate(ys),np.asarray(groups),trips


def build(trial,kind):
    if kind=="LightGBM":
        return LGBMClassifier(objective="binary",
            n_estimators=trial.suggest_int("n_estimators",300,1000),
            learning_rate=trial.suggest_float("learning_rate",0.005,0.08,log=True),
            num_leaves=trial.suggest_int("num_leaves",15,127),
            max_depth=trial.suggest_int("max_depth", -1, 16),
            min_child_samples=trial.suggest_int("min_child_samples",15,100),
            subsample=trial.suggest_float("subsample",0.65,1.0),
            subsample_freq=1,colsample_bytree=trial.suggest_float("colsample_bytree",0.6,1.0),
            reg_lambda=trial.suggest_float("reg_lambda",1e-3,10.0,log=True),
            reg_alpha=trial.suggest_float("reg_alpha",1e-4,2.0,log=True),
            class_weight="balanced",random_state=SEED,n_jobs=-1,verbosity=-1)
    if kind=="Random Forest":
        return make_pipeline(SimpleImputer(strategy="median"),RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators",150,600,step=50),
            max_depth=trial.suggest_int("max_depth",8,32),
            min_samples_split=trial.suggest_int("min_samples_split",2,15),
            min_samples_leaf=trial.suggest_int("min_samples_leaf",1,8),
            max_features=trial.suggest_categorical("max_features",["sqrt",0.5,0.75,1.0]),
            class_weight="balanced_subsample",n_jobs=-1,random_state=SEED))
    if kind=="Logistic Regression":
        return make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(
            C=trial.suggest_float("C",1e-3,100,log=True),max_iter=1500,class_weight="balanced",random_state=SEED))
    if kind=="K-Nearest Neighbours":
        return make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),KNeighborsClassifier(
            n_neighbors=trial.suggest_int("n_neighbors",3,35,step=2),
            weights=trial.suggest_categorical("weights",["uniform","distance"]),
            p=trial.suggest_categorical("p",[1,2]),n_jobs=-1))
    if kind=="Decision Tree":
        return make_pipeline(SimpleImputer(strategy="median"),DecisionTreeClassifier(
            max_depth=trial.suggest_int("max_depth",4,40),
            min_samples_split=trial.suggest_int("min_samples_split",2,20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf",1,12),
            criterion=trial.suggest_categorical("criterion",["gini","entropy","log_loss"]),
            class_weight="balanced",random_state=SEED))
    raise ValueError(f"Unsupported selected classifier: {kind}")


def score(model,trips,held,k):
    correct=total=0
    for t in trips:
        if t["id"] not in held:continue
        p=model.predict_proba(t["X"])[:,1].reshape(t["n"],k)
        cand=t["cids"].reshape(t["n"],k)
        pred=cand[np.arange(t["n"]),np.argmax(p,axis=1)]
        valid=t["target"]>=0
        correct+=int(np.sum((pred==t["target"])&valid));total+=int(valid.sum())
    return correct/max(total,1)


meta=read_model_metadata(MOD)
K=int(meta["k_candidates"])
X,y,groups,trips=collect("train",meta)
selection=json.loads((MOD/"model_selection.json").read_text())
kind=selection["selected_model"]
if kind!=meta.get("selected_classifier"):raise ValueError("Selected classifier metadata mismatch")
cv=GroupKFold(n_splits=min(5,len(np.unique(groups))))
splits=list(cv.split(X,y,groups))

def objective(trial):
    model=build(trial,kind);fold_scores=[]
    for fit_idx,val_idx in splits:
        model.fit(X.iloc[fit_idx],y[fit_idx])
        fold_scores.append(score(model,trips,set(groups[val_idx]),K))
    return float(np.mean(fold_scores))

study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective,n_trials=TRIALS,show_progress_bar=False)
best=build(optuna.trial.FixedTrial(study.best_params),kind)
start=time.perf_counter();best.fit(X,y);fit_s=time.perf_counter()-start
joblib.dump(best,MOD/"final_model.joblib")
meta["selected_classifier"]=kind
meta["tuning_objective"]="5-fold GroupKFold observation-level Top-1 accuracy"
meta["tuning_trials"]=TRIALS
(MOD/"model_metadata.json").write_text(json.dumps(meta,indent=2))
try:
    _,_,_,dev=collect("student_test",meta)
    correct=total=0
    for t in dev:
        p=best.predict_proba(t["X"])[:,1].reshape(t["n"],K);cand=t["cids"].reshape(t["n"],K)
        pred=cand[np.arange(t["n"]),np.argmax(p,axis=1)];valid=t["target"]>=0
        correct+=int(np.sum((pred==t["target"])&valid));total+=int(valid.sum())
    pd.DataFrame([{"model":"Tuned","classifier":kind,"student_test_top1_accuracy":correct/max(total,1)}]).to_csv(OUT/"tuned_vs_untuned.csv",index=False)
except FileNotFoundError:pass
pd.DataFrame([{"trial":t.number,"value":t.value,**t.params} for t in study.trials]).to_csv(OUT/"bayesian_trials.csv",index=False)
summary={"selected_classifier":kind,"sampler":"Optuna TPE",
 "objective":"5-fold GroupKFold observation-level Top-1 accuracy on training trips",
 "trials":TRIALS,"best_params":study.best_params,"best_cv_top1_accuracy":study.best_value,
 "training_trip_count":len(set(groups)),"training_candidate_rows":len(X),
 "candidate_pool":meta["candidate_pool"],"candidate_radius_m":meta["candidate_radius_m"],
 "k_candidates":K,"features":FEATURES,"schema_version":3,"final_fit_seconds":fit_s,
 "evaluator_partition_used":False}
(OUT/"bayesian_best.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
