"""Prediction API; reference routes are excluded from prediction and used only for post-inference aggregate scoring."""
from pathlib import Path
import sys, os, logging
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import joblib
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathsnap.core import read_trip, make_network, candidate_rows, viterbi, FEATURES
from pathsnap.artifacts import read_model_metadata, validate_estimator, read_decoder_metadata

logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
log=logging.getLogger("pathsnap")
app=FastAPI(title="PathSnap API",version="2.0")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
DATA=ROOT/"Student folder"
PARTITION=os.environ.get("PATHSNAP_TRIP_PARTITION","student_test")
MODEL_DIR=ROOT/"pathsnap"/"models"
MODEL_PATH=MODEL_DIR/"final_model.joblib"
MODEL_METADATA=None
model=None
try:
    MODEL_METADATA=read_model_metadata(MODEL_DIR)
    if MODEL_PATH.exists():
        model=validate_estimator(joblib.load(MODEL_PATH),MODEL_METADATA)
except (FileNotFoundError, ValueError) as exc:
    logging.warning("Model artifact is unavailable or stale: %s",exc)
MODEL_K=int(MODEL_METADATA["k_candidates"]) if MODEL_METADATA else 8
CANDIDATE_POOL=int(MODEL_METADATA["candidate_pool"]) if MODEL_METADATA else 64
CANDIDATE_RADIUS_M=float(MODEL_METADATA.get("candidate_radius_m",50.0)) if MODEL_METADATA else 50.0
DECODER_PATH=ROOT/"pathsnap"/"models"/"decoder_metadata.json"
try:
    DECODER_PARAMS=read_decoder_metadata(MODEL_DIR,MODEL_METADATA).get("best_params",{}) if MODEL_METADATA else {}
except (FileNotFoundError,ValueError) as exc:
    logging.warning("Tuned decoder metadata is unavailable or stale; using safe defaults: %s",exc)
    DECODER_PARAMS={}

def partition_dir():
    for p in (DATA/PARTITION, DATA/"train"/PARTITION):
        if p.is_dir(): return p
    raise FileNotFoundError(f"Partition {PARTITION!r} was not found")

@app.get("/health")
def health():
    return {"status":"ok","model_loaded":model is not None,"partition":PARTITION,
            "ground_truth_access":"server-side-metrics-only"}

@app.get("/trips")
def trips():
    try: base=partition_dir()
    except FileNotFoundError as exc: raise HTTPException(404,str(exc))
    return {"trips":[p.name for p in sorted(base.iterdir()) if p.is_dir() and (p/f"{p.name}.track").is_file()]}

@app.post("/predict/{trip_id}")
def predict(trip_id:str):
    if model is None: raise HTTPException(503,"Train the tuned model first: python scripts/tune.py")
    try: folder=partition_dir()/trip_id
    except FileNotFoundError as exc: raise HTTPException(404,str(exc))
    if not folder.is_dir() or not (folder/f"{trip_id}.track").is_file(): raise HTTPException(404,"Unknown trip")
    log.info("Prediction request trip_id=%s partition=%s",trip_id,PARTITION)
    # Critical leakage guard: do not read the reference route in the prediction path.
    track,nodes,arcs,_=read_trip(folder,include_route=False)
    net=make_network(nodes,arcs)
    X,_,candidate_ids,distances,_=candidate_rows(track,net,route=None,k=MODEL_K,radius_m=CANDIDATE_RADIUS_M,candidate_pool=CANDIDATE_POOL)
    probabilities=model.predict_proba(X)[:,1] if hasattr(model,"predict_proba") else model.predict(X).astype(float)*0.98+0.01
    scores=probabilities.reshape(len(track),-1)
    scores=scores/(scores.sum(axis=1,keepdims=True)+1e-9)
    k=scores.shape[1]
    candidate_matrix=candidate_ids.reshape(len(track),k)
    classifier_predicted=candidate_matrix[np.arange(len(track)),np.argmax(scores,axis=1)]
    time_gaps=np.r_[0.0,np.diff(track.timestamp.to_numpy(dtype=float))]
    arc_lengths=np.zeros(len(arcs),dtype=float)
    arc_lengths[net["ids"]]=net["lengths"]
    projection_fractions=X["projection_fraction"].to_numpy(dtype=float)
    gps_displacements=X["gps_displacement_m"].to_numpy(dtype=float).reshape(len(track),k)[:,0]
    predicted=viterbi(candidate_ids,distances,scores.ravel(),arcs,k=k,time_gaps=time_gaps,
                      gap_reset_s=float(os.environ.get("PATHSNAP_GAP_RESET_S",DECODER_PARAMS.get("gap_reset_s",60.0))),
                      projection_fractions=projection_fractions,arc_lengths=arc_lengths,
                      gps_displacements=gps_displacements,beta=DECODER_PARAMS.get("beta",100.0),
                      uturn_penalty=DECODER_PARAMS.get("uturn_penalty",10.0),
                      distance_scale_m=DECODER_PARAMS.get("distance_scale_m",80.0))
    edges=arcs[["from_node","to_node"]].to_numpy(dtype=int)
    good=total=0
    for a,b in zip(predicted[:-1],predicted[1:]):
        if 0<=a<len(edges) and 0<=b<len(edges): total+=1; good+=int(a==b or edges[a,1]==edges[b,0])
    # Evaluation labels are loaded only after inference and never enter the
    # feature matrix, model, decoder or response as raw route/arc labels.
    metrics = None
    try:
        _, _, _, reference_route = read_trip(folder, include_route=True)
        if len(reference_route):
            _, _, eval_candidate_ids, _, eval_target = candidate_rows(
                track, net, route=reference_route, k=MODEL_K, radius_m=CANDIDATE_RADIUS_M, candidate_pool=CANDIDATE_POOL
            )
            valid = eval_target >= 0
            candidate_matrix = eval_candidate_ids.reshape(len(track), -1)
            miss_rate = float(np.mean([
                eval_target[i] not in candidate_matrix[i]
                for i in np.flatnonzero(valid)
            ])) if np.any(valid) else None
            metrics = {
                "reference_observations": int(np.sum(valid)),
                "candidate_miss_rate": miss_rate,
                "classifier_top1_accuracy": float(np.mean(classifier_predicted[valid] == eval_target[valid])) if np.any(valid) else None,
                "decoded_top1_accuracy": float(np.mean(predicted[valid] == eval_target[valid])) if np.any(valid) else None,
            }
    except (FileNotFoundError, ValueError) as exc:
        log.info("Reference metrics unavailable trip_id=%s reason=%s", trip_id, exc)
    route_xy=[]
    for aid in predicted:
        if 0<=int(aid)<len(arcs):
            fr,to=edges[int(aid)]
            if 0<=fr<len(nodes) and 0<=to<len(nodes):
                route_xy.append([[float(nodes.iloc[fr].lon),float(nodes.iloc[fr].lat)],
                                 [float(nodes.iloc[to].lon),float(nodes.iloc[to].lat)]])
    result={"trip_id":trip_id,"partition":PARTITION,"observations":len(track),
            "predicted_arc_ids":predicted.tolist(),"predicted_segments":route_xy,
            "gps":track[["lon","lat"]].values.tolist(),
            "route_continuity":good/max(total,1),
            "accuracy_internal":(metrics["decoded_top1_accuracy"] if metrics else None),
            "evaluation_metrics":metrics}
    log.info("Prediction complete trip_id=%s observations=%d continuity=%.4f",trip_id,len(track),result["route_continuity"])
    return result
