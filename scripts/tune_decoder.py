"""Tune directed-network HMM settings from out-of-fold training predictions only."""
from pathlib import Path
import sys,json,os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from pathsnap.core import trip_features,viterbi,FEATURES
from pathsnap.artifacts import read_model_metadata

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/"Student folder"
MOD=ROOT/"pathsnap"/"models";OUT=ROOT/"pathsnap"/"results"
TRIALS=int(os.environ.get("PATHSNAP_DECODER_TRIALS","40"))


def build_model(kind,p):
    if kind=="LightGBM":
        return LGBMClassifier(objective="binary",class_weight="balanced",random_state=42,n_jobs=-1,verbosity=-1,
            n_estimators=p["n_estimators"],learning_rate=p["learning_rate"],num_leaves=p["num_leaves"],
            max_depth=p["max_depth"],min_child_samples=p["min_child_samples"],subsample=p["subsample"],
            subsample_freq=1,colsample_bytree=p["colsample_bytree"],reg_lambda=p["reg_lambda"],reg_alpha=p["reg_alpha"])
    if kind=="Random Forest":
        return make_pipeline(SimpleImputer(strategy="median"),RandomForestClassifier(
            n_estimators=p["n_estimators"],max_depth=p["max_depth"],min_samples_split=p["min_samples_split"],
            min_samples_leaf=p["min_samples_leaf"],max_features=p["max_features"],
            class_weight="balanced_subsample",n_jobs=-1,random_state=42))
    if kind=="Logistic Regression":
        return make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
            LogisticRegression(C=p["C"],max_iter=1500,class_weight="balanced",random_state=42))
    if kind=="K-Nearest Neighbours":
        return make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
            KNeighborsClassifier(n_neighbors=p["n_neighbors"],weights=p["weights"],p=p["p"],n_jobs=-1))
    if kind=="Decision Tree":
        return make_pipeline(SimpleImputer(strategy="median"),DecisionTreeClassifier(
            max_depth=p["max_depth"],min_samples_split=p["min_samples_split"],min_samples_leaf=p["min_samples_leaf"],
            criterion=p["criterion"],class_weight="balanced",random_state=42))
    raise ValueError(f"Unsupported classifier {kind}")


def collect(meta):
    k=int(meta["k_candidates"]);pool=int(meta["candidate_pool"]);radius=float(meta["candidate_radius_m"])
    xs=[];ys=[];groups=[];trips=[]
    base=DATA/"train"
    if not base.is_dir():base=DATA/"train"/"train"
    folders=[p for p in sorted(base.iterdir()) if p.is_dir() and (p/f"{p.name}.track").is_file()]
    for folder in folders:
        track,nodes,arcs,route,net,X,y,cids,dists,target=trip_features(folder,k=k,candidate_pool=pool,radius_m=radius)
        cand=cids.reshape(len(track),k);hit=np.any(cand==target[:,None],axis=1);keep=np.repeat(hit,k)
        if keep.any():
            xs.append(X.loc[keep].reset_index(drop=True));ys.append(y[keep]);groups.extend([folder.name]*int(keep.sum()))
        lengths=np.zeros(len(arcs),dtype=float);lengths[net["ids"]]=net["lengths"]
        trips.append({"id":folder.name,"track":track,"arcs":arcs,"X":X,"cids":cids,
            "dists":dists,"target":target,"lengths":lengths})
    if not xs:raise RuntimeError(f"No valid training rows in {base}")
    return pd.concat(xs,ignore_index=True),np.concatenate(ys),np.asarray(groups),trips


def score_decoded(t,p,k,params):
    n=len(t["track"])
    p=p/(p.sum(axis=1,keepdims=True)+1e-9)
    gaps=np.r_[0.0,np.diff(t["track"].timestamp.to_numpy(dtype=float))]
    displacement=t["X"]["gps_displacement_m"].to_numpy(dtype=float).reshape(n,k)[:,0]
    decoded=viterbi(t["cids"],t["dists"],p.ravel(),t["arcs"],k=k,time_gaps=gaps,
        gap_reset_s=params["gap_reset_s"],transition_penalty=params["beta"],
        projection_fractions=t["X"]["projection_fraction"].to_numpy(dtype=float),
        arc_lengths=t["lengths"],gps_displacements=displacement,
        speed_weight=params["speed_weight"],max_speed_mps=params["max_speed_mps"],
        distance_scale_m=params["distance_scale_m"],uturn_penalty=params["uturn_penalty"])
    valid=t["target"]>=0
    return int(np.sum((decoded==t["target"])&valid)),int(valid.sum())


meta=read_model_metadata(MOD)
tuning=json.loads((OUT/"bayesian_best.json").read_text())
kind=tuning["selected_classifier"];params=tuning["best_params"]
if kind!=meta.get("selected_classifier"):raise ValueError("Classifier tuning metadata mismatch")
X,y,groups,trips=collect(meta)
cv=GroupKFold(n_splits=min(5,len(np.unique(groups))))
oof={}
for fit_idx,val_idx in cv.split(X,y,groups):
    model=build_model(kind,params);model.fit(X.iloc[fit_idx],y[fit_idx])
    held=set(groups[val_idx])
    for t in trips:
        if t["id"] in held:oof[t["id"]]=model.predict_proba(t["X"])[:,1].reshape(len(t["track"]),int(meta["k_candidates"]))

def objective(trial):
    decoder={"beta":trial.suggest_float("beta",30.0,200.0),
      "gap_reset_s":trial.suggest_float("gap_reset_s",30.0,90.0),
      "uturn_penalty":trial.suggest_float("uturn_penalty",0.0,20.0),
      "speed_weight":trial.suggest_float("speed_weight",0.0,0.08),
      "max_speed_mps":trial.suggest_float("max_speed_mps",25.0,60.0),
      "distance_scale_m":trial.suggest_float("distance_scale_m",20.0,250.0,log=True)}
    correct=total=0
    for t in trips:
        if t["id"] not in oof:continue
        c,n=score_decoded(t,oof[t["id"]],int(meta["k_candidates"]),decoder);correct+=c;total+=n
    return correct/max(total,1)

study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective,n_trials=TRIALS,show_progress_bar=False)
result={"classifier":kind,"method":"5-fold out-of-fold training predictions; no student/evaluator labels",
 "trials":TRIALS,"best_params":study.best_params,"best_oof_top1_accuracy":study.best_value,
 "features":FEATURES,"k_candidates":meta["k_candidates"],"candidate_pool":meta["candidate_pool"],
 "candidate_radius_m":meta["candidate_radius_m"],"schema_version":3}
(MOD/"decoder_metadata.json").write_text(json.dumps(result,indent=2))
pd.DataFrame([{"trial":t.number,"value":t.value,**t.params} for t in study.trials]).to_csv(OUT/"decoder_trials.csv",index=False)
print(json.dumps(result,indent=2))
