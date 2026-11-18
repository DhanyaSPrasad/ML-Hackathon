"""Tune HMM/Viterbi parameters on out-of-fold training-trip predictions only."""
from pathlib import Path
import sys,json,os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,optuna
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from pathsnap.core import trip_features,viterbi,FEATURES
from pathsnap.artifacts import read_model_metadata

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/"Student folder"
MOD=ROOT/"pathsnap"/"models";OUT=ROOT/"pathsnap"/"results"
K=8;POOL=int(os.environ.get("PATHSNAP_CANDIDATE_POOL","64"))
TRIALS=int(os.environ.get("PATHSNAP_DECODER_TRIALS","60"))


def partition_path(name):
    for p in (DATA/name,DATA/"train"/name):
        if p.is_dir():return p
    raise FileNotFoundError(name)


def make_model(kind,p):
    if kind=="Logistic Regression":
        est=LogisticRegression(C=p["C"],max_iter=1500,class_weight="balanced",random_state=42)
        return make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),est)
    if kind=="K-Nearest Neighbours":
        est=KNeighborsClassifier(n_neighbors=p["n_neighbors"],weights=p["weights"],p=p["p"],n_jobs=-1)
        return make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),est)
    if kind=="Decision Tree":
        est=DecisionTreeClassifier(max_depth=p["max_depth"],min_samples_split=p["min_samples_split"],
          min_samples_leaf=p["min_samples_leaf"],criterion=p["criterion"],class_weight="balanced",random_state=42)
        return make_pipeline(SimpleImputer(strategy="median"),est)
    common={k:p[k] for k in ("n_estimators","max_depth","min_samples_split","min_samples_leaf","max_features")}
    common.update(n_jobs=-1,random_state=42,class_weight=p.get("class_weight"))
    est=RandomForestClassifier(**common) if kind=="Random Forest" else ExtraTreesClassifier(**common)
    return make_pipeline(SimpleImputer(strategy="median"),est)


def collect():
    xs=[];ys=[];groups=[];trips=[]
    folders=[p for p in sorted(partition_path("train").iterdir()) if p.is_dir() and (p/f"{p.name}.track").is_file()]
    for folder in folders:
        track,nodes,arcs,route,net,X,y,cids,dists,target=trip_features(folder,k=K,candidate_pool=POOL)
        candidates=cids.reshape(len(track),K);hit=np.any(candidates==target[:,None],axis=1)
        keep=np.repeat(hit,K)
        if keep.any():
            xs.append(X.loc[keep].reset_index(drop=True));ys.append(y[keep]);groups.extend([folder.name]*int(keep.sum()))
        trips.append({"id":folder.name,"track":track,"arcs":arcs,"net":net,"X":X,
          "cids":cids,"dists":dists,"target":target})
    return pd.concat(xs,ignore_index=True),np.concatenate(ys),np.asarray(groups),trips


def movement_inputs(t):
    n=len(t["track"]);k=K;net=t["net"];X=t["X"]
    lengths=np.zeros(len(t["arcs"]),dtype=float);lengths[net["ids"]]=net["lengths"]
    return {"time_gaps":np.r_[0.0,np.diff(t["track"].timestamp.to_numpy(dtype=float))],
      "projection_fractions":X["projection_fraction"].to_numpy(dtype=float),
      "arc_lengths":lengths,
      "gps_displacements":X["gps_displacement_m"].to_numpy(dtype=float).reshape(n,k)[:,0]}


model_metadata=read_model_metadata(MOD)
meta=json.loads((OUT/"bayesian_best.json").read_text())
kind=meta["selected_classifier"];params=meta["best_params"]
if kind != model_metadata.get("selected_classifier"):
    raise ValueError("Classifier tuning result does not match model_metadata.json; rerun scripts/tune.py")
if meta.get("features") != FEATURES or meta.get("features") != model_metadata.get("features"):
    raise ValueError("Classifier tuning feature schema is stale; rerun scripts/train.py and scripts/tune.py")
if int(meta.get("k_candidates", 0)) != int(model_metadata["k_candidates"]) or int(meta.get("candidate_pool", 0)) != int(model_metadata["candidate_pool"]):
    raise ValueError("Classifier tuning candidate configuration differs from model metadata")
if K != int(model_metadata["k_candidates"]) or POOL != int(model_metadata["candidate_pool"]):
    raise ValueError("Decoder tuning candidate settings differ from model metadata; use the same PATHSNAP_CANDIDATE_POOL as training.")
X,y,groups,trips=collect()
cv=GroupKFold(n_splits=min(5,len(np.unique(groups))))
oof={}
for fit_idx,val_idx in cv.split(X,y,groups):
    model=make_model(kind,params);model.fit(X.iloc[fit_idx],y[fit_idx])
    held=set(groups[val_idx])
    for t in trips:
        if t["id"] in held:
            oof[t["id"]]=model.predict_proba(t["X"])[:,1].reshape(len(t["track"]),K)

def objective(trial):
    transition=trial.suggest_float("transition_penalty",0.1,10.0)
    gap=trial.suggest_float("gap_reset_s",2.0,30.0)
    speed=trial.suggest_float("speed_weight",0.0,0.08)
    max_speed=trial.suggest_float("max_speed_mps",25.0,60.0)
    scale=trial.suggest_float("distance_scale_m",20.0,250.0,log=True)
    correct=total=0
    for t in trips:
        if t["id"] not in oof:continue
        n=len(t["track"]);p=oof[t["id"]]
        p=p/(p.sum(axis=1,keepdims=True)+1e-9)
        decoded=viterbi(t["cids"],t["dists"],p.ravel(),t["arcs"],k=K,
          transition_penalty=transition,**movement_inputs(t),gap_reset_s=gap,
          speed_weight=speed,max_speed_mps=max_speed,distance_scale_m=scale)
        valid=t["target"]>=0
        correct+=int(np.sum((decoded==t["target"])&valid));total+=int(valid.sum())
    return correct/max(total,1)

study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective,n_trials=TRIALS,show_progress_bar=False)
result={"classifier":kind,"method":"5-fold out-of-fold training predictions; no student/evaluator test labels",
 "trials":TRIALS,"best_params":study.best_params,"best_oof_top1_accuracy":study.best_value,
 "features":FEATURES,"k_candidates":K,"candidate_pool":POOL,"schema_version":2}
(MOD/"decoder_metadata.json").write_text(json.dumps(result,indent=2))
pd.DataFrame([{"trial":t.number,"value":t.value,**t.params} for t in study.trials]).to_csv(OUT/"decoder_trials.csv",index=False)
print(json.dumps(result,indent=2))
