"""Reproducible benchmarks with metadata, raw folds, repeated sweeps and ablation."""
from pathlib import Path
import argparse
import json
import platform
import hashlib
import numpy as np
import pandas as pd
from data_loader import load_battery_data,dataset_fingerprint
from features import FEATURE_NAMES,FEATURE_GROUPS
from models import ML_MODEL_REGISTRY,METRICS,FACTOR_LEVELS,build_features,fit_lobo_models,evaluate_lobo,simulator_for_factor

SCHEMA_VERSION=2
def code_fingerprint():
    h=hashlib.sha256()
    for name in ['data_loader.py','simulator.py','features.py','models.py','precompute_benchmarks.py']:
        h.update((Path(__file__).parent/name).read_bytes())
    return h.hexdigest()

def run_all_benchmarks(dataset=None,repeats=3,output_dir=None,quick=False):
    dataset=dataset or load_battery_data()
    clean=build_features(dataset)
    benchmarks={};models={};fold_rows=[];predictions=[]
    for label,key in ML_MODEL_REGISTRY.items():
        print('Clean LOBO:',label,flush=True)
        models[key]=fit_lobo_models(dataset,key,clean=clean)
        res=evaluate_lobo(dataset,key,fitted=models[key],clean=clean)
        benchmarks[label]={m:res[m] for m in METRICS}
        fold_rows.extend({'model':label,**f} for f in res['folds'])
        predictions.append(res['predictions'].assign(model=label))
    print('Training augmented RF',flush=True)
    models['rf_aug']=fit_lobo_models(dataset,'rf',augment=True,clean=clean)
    sweep_rows=[];sweep_folds=[]
    for factor,levels in FACTOR_LEVELS.items():
        if quick: levels=[levels[0],levels[-1]]
        for level in levels:
            print('Sweep:',factor,level,flush=True)
            for repeat in range(repeats):
                tests=build_features(dataset,simulator_for_factor(factor,level,42+repeat))
                for key in ['rf','ridge','gb','rf_aug']:
                    res=evaluate_lobo(dataset,fitted=models[key],clean=clean,test_features=tests)
                    base={'factor':factor,'level':level,'repeat':repeat,'model':key}
                    sweep_rows.append({**base,**{m:res[m] for m in METRICS}})
                    sweep_folds.extend({**base,**fold} for fold in res['folds'])
    ablations=[]
    severe=build_features(dataset,simulator_for_factor('charge_window_pct',.5))
    for group,cols in FEATURE_GROUPS.items():
        if group=='Cycle': continue
        retained=[c for c in FEATURE_NAMES if c not in cols]
        fit=fit_lobo_models(dataset,'rf',feature_cols=retained,clean=clean)
        for condition,tests in [('clean',clean),('50% charge',severe)]:
            res=evaluate_lobo(dataset,fitted=fit,clean=clean,test_features=tests)
            ablations.append({'removed_group':group,'condition':condition,**{m:res[m] for m in METRICS}})
    # Held-out permutation importance: no training-set impurity ranking claims.
    importance=[]
    for held,test in clean.items():
        fit=models['rf'][held];base=evaluate_lobo(dataset,fitted={held:fit},clean=clean,test_features={held:test})['soh_mae']
        for col in FEATURE_NAMES:
            for repeat in range(repeats):
                perm=test.copy();perm[col]=np.random.default_rng(700+repeat).permutation(perm[col].values)
                ps,_=fit.predict(perm)
                importance.append({'battery_id':held,'feature':col,'repeat':repeat,
                                   'soh_mae_increase':float(np.abs(test.soh-ps).mean()-base)})
    metadata={'schema_version':SCHEMA_VERSION,'dataset_fingerprint':dataset_fingerprint(dataset),
              'code_fingerprint':code_fingerprint(),'sources':sorted({d['source'] for d in dataset.values()}),
              'battery_count':len(dataset),'cycle_count':sum(len(d['summary']) for d in dataset.values()),
              'threshold':next(iter(dataset.values()))['threshold'],'repeats':repeats,'quick_grid':quick,
              'aggregation':'Equal-weight mean of battery-fold metrics; repeated sensor seeds are not independent batteries.',
              'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__}
    result={'metadata':metadata,'clean_benchmarks':benchmarks,'folds':fold_rows,'sweeps':sweep_rows,
            'sweep_folds':sweep_folds,'ablations':ablations,'permutation_importance':importance}
    def finite(obj):
        if isinstance(obj,dict): return {k:finite(v) for k,v in obj.items()}
        if isinstance(obj,list): return [finite(v) for v in obj]
        if isinstance(obj,float) and not np.isfinite(obj): return None
        return obj
    result=finite(result)
    if output_dir is not None:
        out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
        (out/'benchmark_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        pd.DataFrame(fold_rows).to_csv(out/'clean_folds.csv',index=False)
        pd.concat(predictions).to_csv(out/'clean_predictions.csv',index=False)
        pd.DataFrame(sweep_rows).to_csv(out/'robustness_repeats.csv',index=False)
        pd.DataFrame(sweep_folds).to_csv(out/'robustness_folds.csv',index=False)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',choices=['synthetic','nasa','csv'],default='synthetic')
    p.add_argument('--data-dir');p.add_argument('--output-dir',default='data');p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--quick',action='store_true');args=p.parse_args()
    if args.repeats<1: p.error('repeats must be positive')
    run_all_benchmarks(load_battery_data(source=args.source,data_dir=args.data_dir),args.repeats,args.output_dir,args.quick)
