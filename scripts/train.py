"""Compare the five required classifiers using trip-grouped observation Top-1 CV."""
from pathlib import Path
import os, sys, json, time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss
from pathsnap.core import trip_features, FEATURES, viterbi

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"Student folder"
OUT=ROOT/"pathsnap"/"results"; MOD=ROOT/"pathsnap"/"models"; VIS=ROOT/"pathsnap"/"visuals"
for p in (OUT,MOD,VIS): p.mkdir(parents=True,exist_ok=True)
SEED=42; K=8; CANDIDATE_POOL=int(os.environ.get("PATHSNAP_CANDIDATE_POOL","64"))


def partition_path(name):
    for p in (DATA/name,DATA/"train"/name):
        if p.is_dir(): return p
    raise FileNotFoundError(f"Missing partition {name!r}")


def collect(name):
    base=partition_path(name); rows=[]; targets=[]; groups=[]; trips=[]; ids=[]
    folders=[p for p in sorted(base.iterdir()) if p.is_dir() and (p/f"{p.name}.track").is_file()]
    for folder in folders:
        try:
            track,nodes,arcs,route,net,X,y,cids,dists,target=trip_features(
                folder,k=K,candidate_pool=CANDIDATE_POOL)
            if y is None: continue
            cand=cids.reshape(len(track),K)
            hit=np.any(cand==target[:,None],axis=1)
            # Do not teach the binary classifier that a missing true candidate
            # is a negative example. Keep miss observations for honest Top-1 scoring.
            valid_rows=np.repeat(hit,K)
            if valid_rows.any():
                rows.append(X.loc[valid_rows].reset_index(drop=True))
                targets.append(y[valid_rows])
                groups.extend([folder.name]*int(valid_rows.sum()))
            trips.append({"id":folder.name,"track":track,"arcs":arcs,"X":X,
                          "cids":cids,"dists":dists,"target":target,
                          "miss":~hit,"net":net})
            ids.append(folder.name)
        except Exception as exc:
            print("WARN",name,folder.name,repr(exc),flush=True)
    if not rows: raise RuntimeError(f"No usable trips in {base}")
    return pd.concat(rows,ignore_index=True),np.concatenate(targets),np.asarray(groups),trips,ids


def make_models():
    return {
      "Logistic Regression":make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(max_iter=1000,class_weight="balanced",C=1.0,random_state=SEED)),
      "K-Nearest Neighbours":make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),KNeighborsClassifier(n_neighbors=7,weights="distance")),
      "Decision Tree":make_pipeline(SimpleImputer(strategy="median"),DecisionTreeClassifier(max_depth=12,min_samples_leaf=4,class_weight="balanced",random_state=SEED)),
      "Random Forest":make_pipeline(SimpleImputer(strategy="median"),RandomForestClassifier(n_estimators=200,max_depth=16,min_samples_leaf=3,class_weight="balanced_subsample",n_jobs=-1,random_state=SEED)),
      "Extra Trees":make_pipeline(SimpleImputer(strategy="median"),ExtraTreesClassifier(n_estimators=200,max_depth=18,min_samples_leaf=3,class_weight="balanced",n_jobs=-1,random_state=SEED)),
    }


def candidate_prob(model,X):
    return model.predict_proba(X)[:,1]


def evaluate_top1(model,trips):
    correct=total=0; all_y=[]; all_pred=[]; all_prob=[]
    for trip in trips:
        n=len(trip["track"]); cand=trip["cids"].reshape(n,K)
        p=candidate_prob(model,trip["X"]).reshape(n,K)
        pred=cand[np.arange(n),np.argmax(p,axis=1)]
        target=trip["target"]; valid=target>=0
        correct+=int(np.sum((pred==target)&valid)); total+=int(valid.sum())
        all_y.extend((cand[valid]==target[valid,None]).astype(int).ravel())
        all_pred.extend((np.argmax(p[valid],axis=1)==np.argmax(cand[valid]==target[valid,None],axis=1)).astype(int))
        all_prob.extend(p[valid].ravel())
    return correct/max(total,1),total


def cv_top1(model_template,X,y,groups,trips,cv):
    scores=[]
    for fit_idx,val_idx in cv.split(X,y,groups):
        model=make_models()[model_template]
        model.fit(X.iloc[fit_idx],y[fit_idx])
        held=set(groups[val_idx])
        fold_trips=[t for t in trips if t["id"] in held]
        score,_=evaluate_top1(model,fold_trips); scores.append(score)
    return float(np.mean(scores)),float(np.std(scores))


print("Loading full-resolution training trips...",flush=True)
X,y,groups,train_trips,train_ids=collect("train")
print("Training rows:",len(X),"trips:",len(train_ids),"features:",len(FEATURES),flush=True)
try:
    Xt,yt,gt,dev_trips,dev_ids=collect("student_test")
except FileNotFoundError:
    Xt=yt=gt=None; dev_trips=[]; dev_ids=[]

cv=GroupKFold(n_splits=min(5,len(np.unique(groups))))
rows=[]
for name,model in make_models().items():
    print("Evaluating",name,flush=True)
    start=time.perf_counter(); mean,std=cv_top1(name,X,y,groups,train_trips,cv); cv_s=time.perf_counter()-start
    fit=time.perf_counter(); model.fit(X,y); fit_s=time.perf_counter()-fit
    row={"model":name,"cv_top1_accuracy_mean":mean,"cv_top1_accuracy_std":std,
         "cv_seconds":cv_s,"training_seconds":fit_s,"features":len(FEATURES)}
    if dev_trips:
        top1,_=evaluate_top1(model,dev_trips)
        row["student_test_top1_accuracy"]=top1
    rows.append(row)
    joblib.dump(model,MOD/(name.lower().replace(" ","_")+".joblib"))
    print(name,"CV Top-1",round(mean,4),"+/-",round(std,4),flush=True)

results=pd.DataFrame(rows).sort_values(["cv_top1_accuracy_mean","cv_top1_accuracy_std"],ascending=[False,True])
results.to_csv(OUT/"model_comparison.csv",index=False)
best=results.iloc[0].to_dict()
(MOD/"model_metadata.json").write_text(json.dumps({
 "features":FEATURES,"k_candidates":K,"candidate_pool":CANDIDATE_POOL,
 "observations_per_trip":"full resolution","random_seed":SEED,
 "cv":"5-fold GroupKFold grouped by trip ID",
 "label_definition":"Monotonic dynamic-programming alignment to ordered reference route; weak geometric supervision.",
 "missed_candidate_policy":"Excluded from classifier fitting; counted as Top-1 errors.",
 "evaluator_partition_used":False},indent=2))
(MOD/"model_selection.json").write_text(json.dumps({
 "selection_metric":"mean 5-fold GroupKFold observation-level Top-1 accuracy",
 "selected_model":best["model"],"cv_top1_accuracy_mean":best["cv_top1_accuracy_mean"],
 "cv_top1_accuracy_std":best["cv_top1_accuracy_std"],"candidate_pool":CANDIDATE_POOL,
 "k_candidates":K,"features":FEATURES,"training_trips":len(train_ids),
 "student_test_used_for_selection":False},indent=2))
print("Selected model:",best["model"],flush=True)
