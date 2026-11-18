from pathlib import Path
import sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,joblib
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,log_loss
from pathsnap.core import trip_features,viterbi
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'Student folder';model=joblib.load(ROOT/'pathsnap/models/final_model.joblib');rows=[]; all_y=[]; all_pred=[]; all_prob=[]; all_top=[]
for folder in sorted((DATA/'student_test').iterdir()):
 if not folder.is_dir():continue
 track,nodes,arcs,route,net,X,y,cids,dists,target=trip_features(folder,max_observations=120)
 p=model.predict(X); proba=model.predict_proba(X)[:,1]; n=len(track);k=8
 all_y.extend(y.tolist()); all_pred.extend(p.tolist()); all_prob.extend(proba.tolist())
 # candidate rows are grouped observation by observation
 scores=proba.reshape(n,k);scores=scores/(scores.sum(axis=1,keepdims=True)+1e-9)
 # top candidate arc ids for classifier-only
 cand=cids.reshape(n,k); pred0=cand[np.arange(n),np.argmax(scores,axis=1)]; all_top.extend((y.reshape(n,k)[np.arange(n),np.argmax(scores,axis=1)]==1).tolist())
 pred1=viterbi(cids,dists,scores.ravel(),arcs,k)
 truth=target
 def continuity(a):
  good=total=0
  for i in range(1,len(a)):
   if a[i-1]<len(arcs) and a[i]<len(arcs):
    total+=1;good+=int(arcs.iloc[int(a[i-1]),1]==arcs.iloc[int(a[i]),0])
  return good/max(total,1)
 rows.append({'trip_id':folder.name,'observations':n,'candidate_top1_accuracy':accuracy_score(truth,pred0),'hmm_accuracy':accuracy_score(truth,pred1),'candidate_miss_rate':float(np.mean([truth[i] not in cand[i] for i in range(n)])),'classifier_route_continuity':continuity(pred0),'hmm_route_continuity':continuity(pred1)})
df=pd.DataFrame(rows);df.to_csv(ROOT/'pathsnap/results/route_decoder_comparison.csv',index=False)
summary=df.mean(numeric_only=True).to_dict();(ROOT/'pathsnap/results/route_decoder_summary.json').write_text(pd.Series(summary).to_json(indent=2))
from sklearn.metrics import precision_score,recall_score,f1_score,log_loss
metrics={'candidate_level_accuracy':float(accuracy_score(all_y,all_pred)),'candidate_level_macro_f1':float(f1_score(all_y,all_pred,average='macro',zero_division=0)),'candidate_level_precision':float(precision_score(all_y,all_pred,zero_division=0)),'candidate_level_recall':float(recall_score(all_y,all_pred,zero_division=0)),'candidate_level_log_loss':float(log_loss(all_y,all_prob,labels=[0,1])),'observation_top1_accuracy':float(np.mean(all_top))}
(ROOT/'pathsnap/results/tuned_model_student_metrics.json').write_text(pd.Series(metrics).to_json(indent=2))
print(df.to_string(index=False));print('MEAN',summary);print('TUNED MODEL',metrics)
