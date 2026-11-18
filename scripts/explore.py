from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd,numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from pathsnap.core import read_trip
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'Student folder';OUT=ROOT/'pathsnap/results';OUT.mkdir(exist_ok=True,parents=True)
rows=[];network=[]
for part in ['train','student_test']:
 for folder in sorted((DATA/part).iterdir()):
  if not folder.is_dir():continue
  try:
   t,n,a,r=read_trip(folder); dt=np.diff(t.timestamp.to_numpy()); edges=a.to_numpy(dtype=int); valid=(edges[:,0]>=0)&(edges[:,1]>=0)&(edges[:,0]<len(n))&(edges[:,1]<len(n)); edges=edges[valid]
   graph=coo_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(n),len(n))).tocsr()
   count,labels=connected_components(graph,directed=False); degree=np.asarray(graph.sum(axis=0)).ravel()
   rows.append(dict(partition=part,trip_id=folder.name,gps_points=len(t),nodes=len(n),arcs=len(a),route_arcs=len(r),missing_gps=int(t[['lon','lat','timestamp']].isna().any(axis=1).sum()),time_ordered=bool(t.timestamp.is_monotonic_increasing),duration_s=float(t.timestamp.max()-t.timestamp.min()),max_time_gap_s=float(dt.max()) if len(dt) else 0,irregular_gaps_gt1s=int(np.sum(dt>1.01)),invalid_arc_rows=int((~valid).sum())))
   network.append(dict(partition=part,trip_id=folder.name,connected_components=int(count),isolated_nodes=int(np.sum(degree==0)),largest_component_nodes=int(np.bincount(labels).max()) if len(labels) else 0,valid_directed_arcs=len(edges)))
  except Exception as e: print('SKIP',folder,e)
df=pd.DataFrame(rows);df.to_csv(OUT/'dataset_inventory.csv',index=False);nd=pd.DataFrame(network);nd.to_csv(OUT/'network_connectivity.csv',index=False)
print('Trip counts:',df.groupby('partition').size().to_dict())
print(df.groupby('partition')[['gps_points','nodes','arcs','route_arcs','missing_gps','duration_s','max_time_gap_s','irregular_gaps_gt1s']].agg(['mean','min','max']).to_string())
print('Network connectivity means:',nd.groupby('partition')[['connected_components','isolated_nodes','largest_component_nodes']].mean().to_dict())
