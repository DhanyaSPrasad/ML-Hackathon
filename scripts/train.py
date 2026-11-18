"""Compare candidate classifiers using trip-grouped observation-level Top-1 accuracy."""
from pathlib import Path
import os, sys, json, time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from pathsnap.core import trip_features, FEATURES

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"Student folder"
OUT=ROOT/"pathsnap"/"results"; MOD=ROOT/"pathsnap"/"models"
for p in (OUT,MOD): p.mkdir(parents=True,exist_ok=True)
SEED=42
K=int(os.environ.get("PATHSNAP_K_CANDIDATES","8"))
POOL=int(os.environ.get("PATHSNAP_CANDIDATE_POOL","64"))
RADIUS=float(os.environ.get("PATHSNAP_CANDIDATE_RADIUS_M","50"))


def partition_path(name):
    for p in (DATA/name,DATA/"train"/name):
        if p.is_dir(): return p
    raise FileNotFoundError(f"Missing partition {name!r}")


def collect(name):
    rows=[]; targets=[]; groups=[]; trips=[]; ids=[]
    folders=[p for p in sorted(partition_path(name).iterdir()) if p.is_dir() and (p/f"{p.name}.track").is_file()]
    for folder in folders:
        try:
            track,nodes,arcs,route,net,X,y,cids,dists,target=trip_features(
                folder,k=K,candidate_pool=POOL,radius_m=RADIUS)
            if y is None: continue
            cand=cids.reshape(len(track),K)
            hit=np.any(cand==target[:,None],axis=1)
            keep=np.repeat(hit,K)
            if keep.any():
                rows.append(X.loc[keep].reset_index(drop=True)); targets.append(y[keep])
                groups.extend([folder.name]*int(keep.sum()))
            trips.append({"id":folder.name,"track":track,"arcs":arcs,"X":X,
                          "cids":cids,"dists":dists,"target":target,"net":net})
            ids.append(folder.name)
        except Exception as exc:
            print("WARN",name,folder.name,repr(exc),flush=True)
    if not rows: raise RuntimeError(f"No usable trips in {partition_path(name)}")
    return pd.concat(rows,ignore_index=True),np.concatenate(targets),np.asarray(groups),trips,ids


def make_models():
    return {
      "Logistic Regression":make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(max_iter=1500,class_weight="balanced",C=1.0,random_state=SEED)),
      "K-Nearest Neighbours":make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),KNeighborsClassifier(n_neighbors=7,weights="distance",n_jobs=-1)),
      "Decision Tree":make_pipeline(SimpleImputer(strategy="median"),DecisionTreeClassifier(max_depth=12,min_samples_leaf=4,class_weight="balanced",random_state=SEED)),
      "Random Forest":make_pipeline(SimpleImputer(strategy="median"),RandomForestClassifier(n_estimators=300,max_depth=18,min_samples_leaf=2,class_weight="balanced_subsample",n_jobs=-1,random_state=SEED)),
      "LightGBM":LGBMClassifier(objective="binary",n_estimators=600,learning_rate=0.03,num_leaves=31,
          max_depth=-1,min_child_samples=30,subsample=0.85,colsample_bytree=0.85,
          reg_lambda=1.0,class_weight="balanced",random_state=SEED,n_jobs=-1,verbosity=-1),
    }


def evaluate_top1(model,trips):
    correct=total=misses=0
    for trip in trips:
        n=len(trip["track"]); cand=trip["cids"].reshape(n,K)
        p=model.predict_proba(trip["X"])[:,1].reshape(n,K)
        pred=cand[np.arange(n),np.argmax(p,axis=1)]
        target=trip["target"]; valid=target>=0
        hit=np.any(cand==target[:,None],axis=1)
        misses+=int(np.sum((~hit)&valid)); correct+=int(np.sum((pred==target)&valid)); total+=int(valid.sum())
    return correct/max(total,1),total,misses/max(total,1)


def cv_top1(name,X,y,groups,trips,splits):
    scores=[]
    for fit_idx,val_idx in splits:
        model=make_models()[name]
        model.fit(X.iloc[fit_idx],y[fit_idx])
        held=set(groups[val_idx])
        fold=[t for t in trips if t["id"] in held]
        scores.append(evaluate_top1(model,fold)[0])
    return float(np.mean(scores)),float(np.std(scores))


print("Loading full-resolution training trips...",flush=True)
X,y,groups,train_trips,train_ids=collect("train")
print(f"Training rows: {len(X)} | trips: {len(train_ids)} | features: {len(FEATURES)}",flush=True)
cv=GroupKFold(n_splits=min(5,len(np.unique(groups))))
splits=list(cv.split(X,y,groups))
try:
    _,_,_,dev_trips,_=collect("student_test")
except FileNotFoundError:
    dev_trips=[]
rows=[]
for name in make_models():
    print("Evaluating",name,flush=True)
    start=time.perf_counter(); mean,std=cv_top1(name,X,y,groups,train_trips,splits); elapsed=time.perf_counter()-start
    model=make_models()[name]; fit=time.perf_counter(); model.fit(X,y); fit_s=time.perf_counter()-fit
    row={"model":name,"cv_top1_accuracy_mean":mean,"cv_top1_accuracy_std":std,
         "cv_seconds":elapsed,"training_seconds":fit_s,"features":len(FEATURES)}
    if dev_trips:
        row["student_test_top1_accuracy"],_,row["student_test_candidate_miss_rate"]=evaluate_top1(model,dev_trips)
    rows.append(row)
    artifact=MOD/(name.lower().replace(" ","_")+".joblib")
    joblib.dump(model,artifact)
    print(name,"CV Top-1",round(mean,4),"+/-",round(std,4),flush=True)

results=pd.DataFrame(rows).sort_values(["cv_top1_accuracy_mean","cv_top1_accuracy_std"],ascending=[False,True])
results.to_csv(OUT/"model_comparison.csv",index=False)
best=results.iloc[0].to_dict()
metadata={"schema_version":3,"features":FEATURES,"k_candidates":K,"candidate_pool":POOL,
 "candidate_radius_m":RADIUS,"selected_classifier":best["model"],
 "observations_per_trip":"full resolution","random_seed":SEED,
 "cv":"5-fold GroupKFold grouped by trip ID; observation-level Top-1",
 "label_definition":"Monotonic dynamic-programming alignment to ordered reference route; weak geometric supervision.",
 "missed_candidate_policy":"Excluded from classifier fitting; counted as Top-1 errors.",
 "evaluator_partition_used":False}
(MOD/"model_metadata.json").write_text(json.dumps(metadata,indent=2))
(MOD/"model_selection.json").write_text(json.dumps({
 "selection_metric":"mean 5-fold GroupKFold observation-level Top-1 accuracy",
 "selected_model":best["model"],"cv_top1_accuracy_mean":best["cv_top1_accuracy_mean"],
 "cv_top1_accuracy_std":best["cv_top1_accuracy_std"],"candidate_pool":POOL,
 "candidate_radius_m":RADIUS,"k_candidates":K,"features":FEATURES,
 "training_trips":len(train_ids),"student_test_used_for_selection":False,"schema_version":3},indent=2))
print("Selected model:",best["model"],flush=True)
