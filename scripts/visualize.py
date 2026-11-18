"""Generate model comparison plots and a layered Folium map from student_test."""
from pathlib import Path
import os
import sys
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import joblib
import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from folium import FeatureGroup
from pathsnap.core import read_trip, make_network, candidate_rows, viterbi
from pathsnap.artifacts import read_model_metadata, validate_estimator, read_decoder_metadata
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"Student folder"; RES=ROOT/"pathsnap"/"results"; VIS=ROOT/"pathsnap"/"visuals"
VIS.mkdir(parents=True,exist_ok=True)

def partition_path(name):
    for p in (DATA/name,DATA/"train"/name):
        if p.is_dir(): return p
    raise FileNotFoundError(name)

# Rebuild plots from the actual model-comparison CSV.
metrics=pd.read_csv(RES/"model_comparison.csv")
for metric,title,filename in [
    ("cv_top1_accuracy_mean","Trip-grouped CV observation Top-1 Accuracy","cv_top1_accuracy.png"),
    ("student_test_top1_accuracy","Student-test observation Top-1 Accuracy","student_accuracy.png"),
    ("student_test_candidate_miss_rate","Student-test candidate miss rate","student_candidate_miss_rate.png")]:
    if metric not in metrics: continue
    d=metrics.dropna(subset=[metric]); fig,ax=plt.subplots(figsize=(9,4.8)); ax.bar(d.model,d[metric]); ax.set_title(title); ax.set_ylabel(metric); ax.tick_params(axis="x",rotation=20); fig.tight_layout(); fig.savefig(VIS/filename,dpi=160); plt.close(fig)

metadata=read_model_metadata(ROOT/"pathsnap"/"models")
model=validate_estimator(joblib.load(ROOT/"pathsnap"/"models"/"final_model.joblib"),metadata)
k=int(metadata["k_candidates"])
base=partition_path("student_test")
folders=[p for p in sorted(base.iterdir()) if p.is_dir() and (p/f"{p.name}.track").is_file()]
if not folders: raise RuntimeError("No student_test trips found")
folder=folders[0]
track,nodes,arcs,route=read_trip(folder,include_route=True); net=make_network(nodes,arcs)
decoder=read_decoder_metadata(ROOT/"pathsnap"/"models",metadata).get("best_params",{})
X,_,cids,dists,_=candidate_rows(track,net,route=None,k=k,candidate_pool=int(metadata.get("candidate_pool",64)))
probs=model.predict_proba(X)[:,1]; scores=probs.reshape(len(track),k); scores=scores/(scores.sum(axis=1,keepdims=True)+1e-9)
time_gaps=np.r_[0.0,np.diff(track.timestamp.to_numpy(dtype=float))]
arc_lengths=np.zeros(len(arcs),dtype=float);arc_lengths[net["ids"]]=net["lengths"]
fractions=X["projection_fraction"].to_numpy(dtype=float)
displacement=X["gps_displacement_m"].to_numpy(dtype=float).reshape(len(track),k)[:,0]
pred=viterbi(cids,dists,scores.ravel(),arcs,k=k,time_gaps=time_gaps,
    transition_penalty=decoder.get("transition_penalty",3.0),gap_reset_s=decoder.get("gap_reset_s",5.0),
    projection_fractions=fractions,arc_lengths=arc_lengths,gps_displacements=displacement,
    speed_weight=decoder.get("speed_weight",0.015),max_speed_mps=decoder.get("max_speed_mps",55.0),
    distance_scale_m=decoder.get("distance_scale_m",80.0))
center=[float(track.lat.mean()),float(track.lon.mean())]
m=folium.Map(location=center,zoom_start=14,control_scale=True)
gps_group=FeatureGroup(name="Observed GPS trajectory",show=True)
folium.PolyLine(track[["lat","lon"]].values.tolist(),color="#2878d0",weight=3,opacity=.8,tooltip="Observed GPS").add_to(gps_group); gps_group.add_to(m)
network_group=FeatureGroup(name="Nearby road candidates",show=False)
for aid in np.unique(cids):
    if 0<=int(aid)<len(arcs):
        a,b=arcs.iloc[int(aid)].to_numpy(dtype=int)
        if 0<=a<len(nodes) and 0<=b<len(nodes):
            folium.PolyLine([[nodes.iloc[a].lat,nodes.iloc[a].lon],[nodes.iloc[b].lat,nodes.iloc[b].lon]],color="#9a9a9a",weight=1,opacity=.35).add_to(network_group)
network_group.add_to(m)
ref_group=FeatureGroup(name="Reference route (student-test development only)",show=False)
ref_points=[]
for aid in route:
    if 0<=int(aid)<len(arcs):
        a,b=arcs.iloc[int(aid)].to_numpy(dtype=int)
        if 0<=a<len(nodes) and 0<=b<len(nodes): ref_points.extend([[nodes.iloc[a].lat,nodes.iloc[a].lon],[nodes.iloc[b].lat,nodes.iloc[b].lon]])
if ref_points: folium.PolyLine(ref_points,color="#15945b",weight=4,tooltip="Reference route").add_to(ref_group)
ref_group.add_to(m)
pred_group=FeatureGroup(name="Predicted route (HMM/Viterbi)",show=True)
pred_points=[]
for aid in pred:
    if 0<=int(aid)<len(arcs):
        a,b=arcs.iloc[int(aid)].to_numpy(dtype=int)
        if 0<=a<len(nodes) and 0<=b<len(nodes): pred_points.extend([[nodes.iloc[a].lat,nodes.iloc[a].lon],[nodes.iloc[b].lat,nodes.iloc[b].lon]])
if pred_points: folium.PolyLine(pred_points,color="#e05a32",weight=5,tooltip="Predicted route").add_to(pred_group)
pred_group.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)
folium.Marker([track.iloc[0].lat,track.iloc[0].lon],tooltip="Start",icon=folium.Icon(color="green")).add_to(m)
folium.Marker([track.iloc[-1].lat,track.iloc[-1].lon],tooltip="End",icon=folium.Icon(color="red")).add_to(m)
html=VIS/f"trip_{folder.name}_map.html"; m.save(html)

print("Generated model plots and",html)
