"""Export trained estimators and exact sensor arrays for browser inference."""
from pathlib import Path
import sys,json,shutil,argparse
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python_app'))
from data_loader import load_battery_data
from models import fit_lobo_models,build_features,ML_MODEL_REGISTRY
from features import extract_cycle_features
from simulator import DataQualitySimulator

def tree(t):
    t=t.tree_
    return {'feature':t.feature.tolist(),'threshold':t.threshold.tolist(),
            'left':t.children_left.tolist(),'right':t.children_right.tolist(),
            'value':t.value[:,0,0].tolist()}

def estimator(m):
    if hasattr(m,'coef_'): return {'kind':'linear','coef':m.coef_.tolist(),'intercept':float(m.intercept_)}
    if hasattr(m,'learning_rate'):
        return {'kind':'gb','base':float(m.init_.constant_[0,0]),'rate':m.learning_rate,
                'trees':[tree(t[0]) for t in m.estimators_]}
    return {'kind':'rf','trees':[tree(t) for t in m.estimators_]}

def dump(path,value):
    path.write_text(json.dumps(value,separators=(',',':'),allow_nan=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--results-dir',required=True);args=p.parse_args()
    ds=load_battery_data(source='nasa',data_dir=args.data_dir);clean=build_features(ds);out=ROOT/'browser_app/data'
    for item in (out/'raw').glob('*.bin'):item.unlink()
    manifest={'source':'NASA PCoE original MATLAB measurements','batteries':{},'models':ML_MODEL_REGISTRY,'default_battery':'B0005','default_cycle':84}
    for bid,d in ds.items():
        manifest['batteries'][bid]={'summary':d['summary'][['cycle','soh','rul','capacity_ah']].to_dict('records'),
          'initial_capacity':d['specs']['c0'],'eol_cycle':d['eol_cycle']}
        for k,trace in d['timeseries'].items():
            trace[['time','voltage','current','temperature']].to_numpy(dtype='<f8').tofile(out/'raw'/f'{bid}-{k}.bin')
    dump(out/'manifest.json',manifest)
    validation=[]
    for augmented in [False,True]:
        for label,key in ML_MODEL_REGISTRY.items():
            print('Export',key,'augmented' if augmented else 'clean',flush=True)
            models=fit_lobo_models(ds,key,augment=augmented,clean=clean)
            for bid,m in models.items():
                serialized={'features':m.feature_cols,'impute':m.imputer.statistics_.tolist(),
                   'mean':m.scaler.mean_.tolist(),'scale':m.scaler.scale_.tolist(),
                   'soh':estimator(m.model_soh),'rul':estimator(m.model_rul) if m.has_rul else None,
                   'trained_on':[b for b in ds if b!=bid],'augmentation':augmented}
                name=f'{bid}-{key}-'+('aug' if augmented else 'clean')+'.json'
                dump(out/'models'/name,serialized)
                k=int(ds[bid]['summary'].cycle.iloc[len(ds[bid]['summary'])//2])
                for config in [{},{'charge_window_pct':.5},{'noise_std_voltage':.05,'missing_rate':.5}]:
                    trace=DataQualitySimulator(**config).apply(ds[bid]['timeseries'][k])
                    f=extract_cycle_features(trace,k)
                    ps,pr=m.predict(pd.DataFrame([f]))
                    validation.append({'model':name,'features':{n:(float(v) if np.isfinite(v) else None) for n,v in f.items()},
                                       'expected':[float(ps[0]),float(pr[0])]})
    dump(ROOT/'browser_tests/browser_model_fixtures.json',validation)
    fixtures=[]
    for cfg in [{},{'charge_window_pct':.5,'charge_window_start':.5},{'missing_rate':.7,'burst_drop_gap':True},{'sampling_step':30,'noise_std_voltage':.05}]:
        trace=DataQualitySimulator(**cfg).apply(ds['B0005']['timeseries'][84])
        f=extract_cycle_features(trace,84)
        fixtures.append({'trace':trace[['time','voltage','current','temperature']].values.tolist(),
                         'features':{n:(float(v) if np.isfinite(v) else None) for n,v in f.items()}})
    dump(ROOT/'browser_tests/browser_feature_fixtures.json',fixtures)
    shutil.copy2(Path(args.results_dir)/'benchmark_summary.json',out/'benchmarks.json')
    print('Export complete:',len(validation),'prediction fixtures;',len(fixtures),'feature fixtures',flush=True)

if __name__=='__main__': main()
