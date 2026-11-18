"""Training-only candidate radius/count sweep; does not inspect student/evaluator partitions."""
from pathlib import Path
import os,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from pathsnap.core import read_trip,make_network,candidate_rows

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/"Student folder"
OUT=ROOT/"pathsnap"/"results";OUT.mkdir(parents=True,exist_ok=True)
RADII=[20,30,50,75,100,150]
COUNTS=[4,8,12,16]
POOL=max(128,int(os.environ.get("PATHSNAP_CANDIDATE_POOL","128")))
base=DATA/"train"
if not base.is_dir():base=DATA/"train"/"train"
folders=[p for p in sorted(base.iterdir()) if p.is_dir() and (p/f"{p.name}.track").is_file()]
if not folders:raise RuntimeError(f"No training trips found in {base}")
rows=[]
for folder in folders:
    track,nodes,arcs,route=read_trip(folder,include_route=True)
    net=make_network(nodes,arcs)
    # Align targets once; targets do not depend on candidate radius/count.
    _,_,_,_,target=candidate_rows(track,net,route=route,k=1,radius_m=150,candidate_pool=POOL)
    for radius in RADII:
        for k in COUNTS:
            _,_,ids,_,_=candidate_rows(track,net,route=None,k=k,radius_m=radius,candidate_pool=POOL)
            cand=ids.reshape(len(track),k)
            valid=target>=0
            miss=np.asarray([target[i] not in cand[i] for i in range(len(track))])
            rows.append({"trip_id":folder.name,"radius_m":radius,"k_candidates":k,
                "observations":int(valid.sum()),
                "candidate_miss_rate":float(miss[valid].mean()) if valid.any() else np.nan})
    print("Completed",folder.name,flush=True)
df=pd.DataFrame(rows)
df["misses_estimate"]=df["candidate_miss_rate"]*df["observations"]
summary=df.groupby(["radius_m","k_candidates"]).agg(
    trips=("trip_id","nunique"),observations=("observations","sum"),
    mean_trip_candidate_miss_rate=("candidate_miss_rate","mean"),
    misses_estimate=("misses_estimate","sum")).reset_index()
summary["pooled_candidate_miss_rate"]=summary["misses_estimate"]/summary["observations"]
summary=summary.drop(columns=["misses_estimate"])
df.to_csv(OUT/"candidate_sweep_by_trip.csv",index=False)
summary.to_csv(OUT/"candidate_sweep_summary.csv",index=False)
print(summary.sort_values("pooled_candidate_miss_rate").to_string(index=False))
