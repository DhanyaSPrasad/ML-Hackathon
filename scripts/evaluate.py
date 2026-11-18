"""Compare classifier-only and movement-aware Viterbi on student_test."""
from pathlib import Path
import os,sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,joblib
from sklearn.metrics import accuracy_score
from pathsnap.core import trip_features,viterbi

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/"Student folder"
OUT=ROOT/"pathsnap"/"results";MODEL_DIR=ROOT/"pathsnap"/"models";OUT.mkdir(parents=True,exist_ok=True)
K=8;POOL=int(os.environ.get("PATHSNAP_CANDIDATE_POOL","64"))
try:
    DECODER=json.loads((MODEL_DIR/"decoder_metadata.json").read_text()).get("best_params",{})
except (FileNotFoundError,ValueError):
    DECODER={}

def partition_path(name):
    for p in (DATA/name,DATA/"train"/name):
        if p.is_dir():return p
    raise FileNotFoundError(name)

def continuity(sequence,arcs):
    edges=arcs[["from_node","to_node"]].to_numpy(dtype=int);good=total=0
    for a,b in zip(sequence[:-1],sequence[1:]):
        if 0<=a<len(edges) and 0<=b<len(edges):
            total+=1;good+=int(a==b or edges[a,1]==edges[b,0])
    return good/max(total,1)

folders=[p for p in sorted(partition_path("student_test").iterdir()) if p.is_dir() and (p/f"{p.name}.track").is_file()]
models=[(p.stem.replace("_"," ").title(),joblib.load(p)) for p in sorted(MODEL_DIR.glob("*.joblib"))]
if not folders:raise RuntimeError("No student_test trips found")
if not models:raise RuntimeError("No saved models found")
rows=[]
for name,model in models:
    for folder in folders:
        track,nodes,arcs,route,net,X,y,cids,dists,target=trip_features(folder,k=K,candidate_pool=POOL)
        n=len(track);prob=model.predict_proba(X)[:,1].reshape(n,K)
        prob=prob/(prob.sum(axis=1,keepdims=True)+1e-9)
        cand=cids.reshape(n,K);classifier=cand[np.arange(n),np.argmax(prob,axis=1)]
        gaps=np.r_[0.0,np.diff(track.timestamp.to_numpy(dtype=float))]
        arc_lengths=np.zeros(len(arcs),dtype=float);arc_lengths[net["ids"]]=net["lengths"]
        fractions=X["projection_fraction"].to_numpy(dtype=float)
        displacement=X["gps_displacement_m"].to_numpy(dtype=float).reshape(n,K)[:,0]
        decoded=viterbi(cids,dists,prob.ravel(),arcs,k=K,time_gaps=gaps,
          gap_reset_s=float(os.environ.get("PATHSNAP_GAP_RESET_S",DECODER.get("gap_reset_s",5.0))),
          transition_penalty=DECODER.get("transition_penalty",3.0),
          projection_fractions=fractions,arc_lengths=arc_lengths,gps_displacements=displacement,
          speed_weight=DECODER.get("speed_weight",0.015),
          max_speed_mps=DECODER.get("max_speed_mps",55.0),
          distance_scale_m=DECODER.get("distance_scale_m",80.0))
        valid=target>=0
        miss=np.asarray([target[i] not in cand[i] for i in range(n)])
        rows.append({"model":name,"trip_id":folder.name,"observations":n,
          "candidate_miss_rate":float(miss[valid].mean()) if valid.any() else None,
          "classifier_top1_accuracy":float(accuracy_score(target[valid],classifier[valid])) if valid.any() else None,
          "hmm_accuracy":float(accuracy_score(target[valid],decoded[valid])) if valid.any() else None,
          "classifier_route_continuity":continuity(classifier,arcs),
          "hmm_route_continuity":continuity(decoded,arcs)})
df=pd.DataFrame(rows);df.to_csv(OUT/"route_decoder_comparison.csv",index=False)
summary=df.groupby("model").mean(numeric_only=True).to_dict(orient="index")
(OUT/"route_decoder_summary.json").write_text(json.dumps(summary,indent=2))
print(df.groupby("model")[["classifier_top1_accuracy","hmm_accuracy","classifier_route_continuity","hmm_route_continuity"]].mean().to_string())
