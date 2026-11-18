from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd,matplotlib.pyplot as plt,folium
from pathsnap.core import read_trip
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'pathsnap/visuals';OUT.mkdir(exist_ok=True,parents=True)
df=pd.read_csv(ROOT/'pathsnap/results/model_comparison.csv')
for metric,title,file in [('cv_macro_f1','Group-aware CV Macro F1','cv_macro_f1.png'),('student_top1_accuracy','Student-test Top-1 Accuracy','student_accuracy.png'),('student_macro_f1','Student-test Macro F1','student_macro_f1.png'),('student_log_loss','Student-test Log Loss','student_log_loss.png')]:
 d=df.dropna(subset=[metric]);plt.figure(figsize=(9,4.8));plt.bar(d.model,d[metric]);plt.title(title);plt.ylabel(metric);plt.xticks(rotation=20,ha='right');plt.tight_layout();plt.savefig(OUT/file,dpi=160);plt.close()
# Actual trajectory map for a supplied development trip
folder=ROOT/'Student folder/student_test/00000000';track,nodes,arcs,route=read_trip(folder);center=[float(track.lat.mean()),float(track.lon.mean())]
m=folium.Map(location=center,zoom_start=13,control_scale=True);folium.PolyLine(track[['lat','lon']].values.tolist(),color='blue',weight=3,tooltip='Observed GPS track').add_to(m)
# Reference route is used only in this offline development visualisation, not by the prediction UI.
pts=[]
for aid in route:
 if 0<=aid<len(arcs):
  a,b=arcs.iloc[int(aid)].to_numpy(dtype=int)
  if a<len(nodes) and b<len(nodes):pts.extend([[float(nodes.iloc[a].lat),float(nodes.iloc[a].lon)],[float(nodes.iloc[b].lat),float(nodes.iloc[b].lon)]])
if pts:folium.PolyLine(pts,color='green',weight=4,tooltip='Reference route (offline development EDA)').add_to(m)
folium.LayerControl().add_to(m);m.save(OUT/'trip_00000000_eda.html')
print('Wrote plots and Folium map to',OUT)
