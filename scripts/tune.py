"""Compact Bayesian optimisation run; uses only the 80 training trips."""
from pathlib import Path
import sys,json,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,joblib,optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold,cross_val_score
from pathsnap.core import trip_features,FEATURES
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'Student folder'; OUT=ROOT/'pathsnap/results'; MOD=ROOT/'pathsnap/models'
Xs=[];ys=[];groups=[]
for folder in sorted((DATA/'train').iterdir()):
 if folder.is_dir():
  try:
   *_,x,y,_,_,_=trip_features(folder,max_observations=40);Xs.append(x);ys.append(y);groups.extend([folder.name]*len(y))
  except Exception as e:print('skip',folder.name,e)
X=pd.concat(Xs,ignore_index=True);y=np.concatenate(ys);g=np.asarray(groups)
print('Bayesian search data',X.shape,'groups',len(np.unique(g)),flush=True)
def objective(trial):
 m=make_pipeline(SimpleImputer(),RandomForestClassifier(n_estimators=trial.suggest_int('n_estimators',20,40),max_depth=trial.suggest_int('max_depth',8,16),min_samples_leaf=trial.suggest_int('min_samples_leaf',2,6),max_features=trial.suggest_categorical('max_features',['sqrt',0.8]),class_weight='balanced_subsample',n_jobs=2,random_state=42))
 return float(cross_val_score(m,X,y,groups=g,cv=GroupKFold(2),scoring='f1_macro',n_jobs=1).mean())
study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=42));study.optimize(objective,n_trials=3)
model=make_pipeline(SimpleImputer(),RandomForestClassifier(**study.best_params,class_weight='balanced_subsample',n_jobs=2,random_state=42));t=time.time();model.fit(X,y);joblib.dump(model,MOD/'final_model.joblib')
pd.DataFrame([{'trial':x.number,'value':x.value,'params':json.dumps(x.params)} for x in study.trials]).to_csv(OUT/'bayesian_trials.csv',index=False)
(OUT/'bayesian_best.json').write_text(json.dumps({'best_params':study.best_params,'best_cv_macro_f1':study.best_value,'search_training_samples':len(X),'group_count':len(np.unique(g)),'trials':len(study.trials),'refit_seconds':time.time()-t,'final_model_fit_samples':len(X),'final_model_fit_observations_per_trip':40},indent=2))
print('Best',study.best_params,study.best_value,flush=True)
