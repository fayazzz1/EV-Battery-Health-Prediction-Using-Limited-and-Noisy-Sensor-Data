"""Battery-disjoint evaluation, train-only imputation and repeatable stress tests."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from features import FEATURE_NAMES, extract_features_for_battery
from simulator import DataQualitySimulator

ML_MODEL_REGISTRY={'Cycle Count Baseline':'cycle_only','Ridge Regression':'ridge','Random Forest':'rf','Gradient Boosting':'gb'}
METRICS=['soh_mae','soh_rmse','rul_mae','rul_rmse']

class BatteryHealthModelWrapper:
    def __init__(self,model_type='rf',feature_cols=None):
        self.model_type=model_type
        self.feature_cols=['cycle'] if model_type=='cycle_only' else list(feature_cols or FEATURE_NAMES)
        self.imputer=SimpleImputer(strategy='median',keep_empty_features=True)
        self.scaler=StandardScaler()
        def make():
            if model_type=='cycle_only': return LinearRegression()
            if model_type=='ridge': return Ridge(alpha=10.)
            if model_type=='rf': return RandomForestRegressor(n_estimators=80,max_depth=8,min_samples_leaf=2,random_state=42,n_jobs=2)
            if model_type=='gb': return GradientBoostingRegressor(n_estimators=80,max_depth=3,learning_rate=.08,random_state=42)
            raise ValueError('Unknown model: '+model_type)
        self.model_soh=make();self.model_rul=make();self.has_rul=False

    def fit(self,X,y_soh,y_rul):
        values=self.imputer.fit_transform(X[self.feature_cols])
        values=self.scaler.fit_transform(values)
        self.model_soh.fit(values,y_soh)
        valid=np.isfinite(y_rul);self.has_rul=bool(valid.any())
        if self.has_rul: self.model_rul.fit(values[valid],np.asarray(y_rul)[valid])
        return self

    def predict(self,X):
        values=self.scaler.transform(self.imputer.transform(X[self.feature_cols]))
        soh=self.model_soh.predict(values)
        rul=np.maximum(0,self.model_rul.predict(values)) if self.has_rul else np.full(len(X),np.nan)
        return soh,rul

def score(y,p):
    y,p=np.asarray(y),np.asarray(p);valid=np.isfinite(y)&np.isfinite(p)
    if not valid.any(): return np.nan,np.nan
    err=y[valid]-p[valid]
    return float(np.abs(err).mean()),float(np.sqrt(np.square(err).mean()))

def build_features(dataset,simulator=None):
    return {bid:extract_features_for_battery(data,simulator) for bid,data in dataset.items()}

def fit_lobo_models(dataset,model_type='rf',augment=False,feature_cols=None,clean=None):
    if len(dataset)<2: raise ValueError('At least two independent batteries required for LOBO.')
    clean=clean or build_features(dataset)
    augmented=build_features(dataset,DataQualitySimulator(sampling_step=5,noise_std_voltage=.015,
        noise_std_current=.03,noise_std_temp=.4,missing_rate=.2,charge_window_pct=.7,seed=911)) if augment else None
    models={}
    for held in dataset:
        ids=[bid for bid in dataset if bid!=held]
        parts=[clean[bid] for bid in ids]
        if augment: parts.extend(augmented[bid] for bid in ids)
        train=pd.concat(parts,ignore_index=True)
        models[held]=BatteryHealthModelWrapper(model_type,feature_cols).fit(train,train.soh.to_numpy(),train.rul.to_numpy())
    return models

def evaluate_lobo(dataset,model_type='rf',test_simulator=None,augment=False,feature_cols=None,fitted=None,clean=None,test_features=None):
    clean=clean or build_features(dataset)
    fitted=fitted or fit_lobo_models(dataset,model_type,augment,feature_cols,clean)
    tests=test_features or (build_features(dataset,test_simulator) if test_simulator else clean)
    folds=[];predictions=[]
    for held,test in tests.items():
        ps,pr=fitted[held].predict(test)
        sm,sr=score(test.soh,ps);rm,rr=score(test.rul,pr)
        folds.append({'test_battery':held,'train_batteries':', '.join(b for b in dataset if b!=held),
                      'n_cycles':len(test),'n_rul_scored':int((np.isfinite(test.rul)&np.isfinite(pr)).sum()),
                      'soh_mae':sm,'soh_rmse':sr,'rul_mae':rm,'rul_rmse':rr})
        pred=test[['battery_id','cycle','soh','rul']].copy();pred['pred_soh']=ps;pred['pred_rul']=pr;predictions.append(pred)
    predictions=pd.concat(predictions,ignore_index=True)
    # Equal weight per independent battery is the primary result.
    result={m:float(pd.Series([f[m] for f in folds]).mean()) for m in METRICS}
    a,b=score(predictions.soh,predictions.pred_soh);c,d=score(predictions.rul,predictions.pred_rul)
    result.update(folds=folds,predictions=predictions,pooled={'soh_mae':a,'soh_rmse':b,'rul_mae':c,'rul_rmse':d})
    return result

FACTOR_LEVELS={'sampling_step':[1,2,5,10,20,30], 'noise_std_voltage':[0,.005,.015,.03,.05,.1],
               'missing_rate':[0,.05,.15,.3,.5,.7], 'charge_window_pct':[1,.85,.7,.5,.35]}

def simulator_for_factor(factor,level,seed=42):
    if factor=='noise_std_voltage':
        return DataQualitySimulator(noise_std_voltage=level,noise_std_current=level*2,noise_std_temp=level*25,seed=seed)
    return DataQualitySimulator(**{factor:level},seed=seed)

def run_robustness_sweep(dataset,factor_name,levels,model_type='rf',repeats=3,augment=False):
    clean=build_features(dataset);models=fit_lobo_models(dataset,model_type,augment,clean=clean);rows=[]
    for level in levels:
        for repeat in range(repeats):
            result=evaluate_lobo(dataset,model_type,simulator_for_factor(factor_name,level,42+repeat),fitted=models,clean=clean)
            rows.append({'factor':factor_name,'level':level,'repeat':repeat,**{m:result[m] for m in METRICS}})
    return pd.DataFrame(rows)

def quality_boundary(frame,tolerance,baseline,metric='soh_mae',reverse=False):
    """First crossing along clean-to-severe grid; no interpolation or extrapolation.

    Non-monotonic later recoveries do not erase an earlier failure.
    """
    ordered=frame.groupby('level',as_index=False)[metric].mean().sort_values('level',ascending=not reverse)
    last=None;failure=None;break_even=None
    for _,row in ordered.iterrows():
        if failure is None:
            if not np.isfinite(row[metric]) or row[metric]>tolerance: failure=float(row.level)
            else: last=float(row.level)
        if break_even is None and np.isfinite(row[metric]) and row[metric]>=baseline: break_even=float(row.level)
    return {'last_passing_level':last,'first_failing_level':failure,'first_not_better_than_baseline':break_even}
