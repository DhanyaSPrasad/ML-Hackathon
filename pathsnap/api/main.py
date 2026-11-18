from pathlib import Path
import sys,logging
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,joblib,os
from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathsnap.core import read_trip,make_network,candidate_rows,viterbi
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s');log=logging.getLogger('pathsnap')
app=FastAPI(title='PathSnap API',version='1.0')
app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['*'],allow_headers=['*'])
DATA=ROOT/'Student folder'; PARTITION=os.environ.get('PATHSNAP_TRIP_PARTITION','student_test'); MODEL_PATH=ROOT/'pathsnap/models/final_model.joblib'
model=joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
@app.get('/health')
def health(): return {'status':'ok','model_loaded':model is not None}
@app.get('/trips')
def trips():
 p=DATA/PARTITION
 return {'trips':[x.name for x in sorted(p.iterdir()) if x.is_dir()]}
@app.post('/predict/{trip_id}')
def predict(trip_id:str):
 if model is None: raise HTTPException(503,'Train the model first: python scripts/train.py')
 folder=DATA/PARTITION/trip_id
 if not folder.exists(): raise HTTPException(404,'Unknown trip')
 log.info('Inference request trip_id=%s',trip_id)
 track,nodes,arcs,route=read_trip(folder);net=make_network(nodes,arcs)
 X,_,cids,dists,target=candidate_rows(track,net,route,k=8)
 proba=model.predict_proba(X)[:,1] if hasattr(model,'predict_proba') else model.predict(X).astype(float)*.98+.01
 # Scores are normalised per observation across the candidate set.
 scores=proba.reshape(len(track),-1); scores=scores/(scores.sum(axis=1,keepdims=True)+1e-9)
 k=scores.shape[1]; pred=viterbi(cids,dists,scores.ravel(),arcs,k=k)
 # Internal reference evaluation; route labels never leave the API except aggregate metrics.
 truth=target
 acc=float(np.mean(pred==truth)) if len(truth) else None
 continuity=float(np.mean([arcs.iloc[int(pred[i-1])].to_numpy()[1]==arcs.iloc[int(pred[i])].to_numpy()[0] for i in range(1,len(pred)) if pred[i-1]<len(arcs) and pred[i]<len(arcs)])) if len(pred)>1 else 0.0
 route_xy=[]
 for aid in pred:
  if 0<=int(aid)<len(arcs):
   fr,to=arcs.iloc[int(aid)].to_numpy(dtype=int)
   if fr<len(nodes) and to<len(nodes): route_xy.append([[float(nodes.iloc[fr].lon),float(nodes.iloc[fr].lat)],[float(nodes.iloc[to].lon),float(nodes.iloc[to].lat)]])
 result={'trip_id':trip_id,'observations':len(track),'predicted_arc_ids':pred.tolist(),'predicted_segments':route_xy,'gps':track[['lon','lat']].values.tolist(),'accuracy_internal':acc,'route_continuity':continuity}
 log.info('Completed trip_id=%s observations=%d internal_accuracy=%s continuity=%.4f',trip_id,len(track),acc,continuity)
 return result
