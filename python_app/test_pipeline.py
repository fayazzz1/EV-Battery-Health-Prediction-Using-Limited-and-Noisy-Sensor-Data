import io
import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat
from data_loader import label_battery,load_nasa_mat,load_sensor_csv,generate_nasa_dataset,dataset_fingerprint
from simulator import DataQualitySimulator
from features import extract_cycle_features,extract_features_for_battery,FEATURE_NAMES
from models import evaluate_lobo,fit_lobo_models,quality_boundary,build_features

def trace():
    return pd.DataFrame({'time':np.arange(101.),'voltage':np.linspace(3.6,4.2,101),
                         'current':np.full(101,1.5),'temperature':np.linspace(24,28,101)})

def tiny():
    result={}
    for b in ['a','b','c']:
        rows=pd.DataFrame({'battery_id':b,'cycle':[1,2,3,4],'capacity_ah':[2,1.9,1.6,1.5]})
        result[b]=label_battery(rows,{i:trace() for i in range(1,5)},'fixture')
    return result

def test_labels_and_censoring():
    ds=tiny();s=ds['a']['summary']
    assert s.soh.tolist()==[100,95,80,75]
    assert s.rul.tolist()==[2,1,0,0]
    late=label_battery(s[['battery_id','cycle','capacity_ah']].iloc[:2],{},'fixture')
    assert late['eol_cycle'] is None
    assert late['summary'].rul.isna().all()

def test_synthetic_reproducible_and_labeled():
    a=generate_nasa_dataset();b=generate_nasa_dataset()
    assert dataset_fingerprint(a)==dataset_fingerprint(b)
    assert all(d['source']=='Synthetic demonstration' for d in a.values())
    assert all(d['summary'].soh.iloc[0]==100 for d in a.values())

def test_simulator_does_not_mutate_and_upper_window():
    raw=trace();copy=raw.copy()
    sim=DataQualitySimulator(charge_window_pct=.5,charge_window_start=.5,sampling_step=2)
    degraded=sim.apply(raw)
    pd.testing.assert_frame_equal(raw,copy)
    assert degraded.time.iloc[0]==50
    assert len(degraded)==25

def test_degenerate_features_and_charge_integration():
    f=extract_cycle_features(trace(),1)
    assert f['coulomb_capacity_ah']==pytest.approx(1.5*100/3600)
    assert np.isnan(extract_cycle_features(trace().iloc[:1],1)['v_mean'])
    assert np.isnan(extract_cycle_features(trace().iloc[50:],1)['v_trans_time_sec'])

def test_gap_is_not_integrated():
    t=trace();g=t[(t.time<20)|(t.time>80)]
    assert extract_cycle_features(g,1)['coulomb_capacity_ah']<.02

def test_repeatable_per_cycle_degradation_and_fixed_labels():
    ds=tiny();sim=DataQualitySimulator(noise_std_voltage=.1,missing_rate=.3)
    a=extract_features_for_battery(ds['a'],sim);b=extract_features_for_battery(ds['a'],sim)
    pd.testing.assert_frame_equal(a,b)
    assert a.soh.tolist()==ds['a']['summary'].soh.tolist()

def test_lobo_and_fold_local_imputation():
    ds=tiny();clean=build_features(ds)
    clean['a']['v_mean']=999
    models=fit_lobo_models(ds,'ridge',clean=clean)
    idx=models['a'].feature_cols.index('v_mean')
    assert models['a'].imputer.statistics_[idx]<5
    res=evaluate_lobo(ds,'ridge',clean=clean,fitted=models)
    assert len(res['folds'])==3
    assert all(f['test_battery'] not in f['train_batteries'].split(', ') for f in res['folds'])
    assert res['soh_mae']==pytest.approx(np.mean([f['soh_mae'] for f in res['folds']]))

def test_no_eol_rul_not_fabricated():
    ds=tiny()
    for d in ds.values(): d['summary']['rul']=np.nan
    result=evaluate_lobo(ds,'ridge')
    assert np.isnan(result['rul_mae'])
    assert result['predictions'].pred_rul.isna().all()

def test_sampled_quality_boundary():
    df=pd.DataFrame({'level':[1,2,5,10],'soh_mae':[1,2,4,1]})
    q=quality_boundary(df,3,3.5)
    assert q=={'last_passing_level':2.,'first_failing_level':5.,'first_not_better_than_baseline':5.}
    reverse=pd.DataFrame({'level':[1,.5,.3],'soh_mae':[1,4,5]})
    assert quality_boundary(reverse,3,4,reverse=True)['last_passing_level']==1

def test_csv_import_and_validation():
    frames=[]
    for k,cap in [(1,2),(2,1.5)]: frames.append(trace().assign(battery_id='cell',cycle=k,capacity_ah=cap))
    ds=load_sensor_csv(io.StringIO(pd.concat(frames).to_csv(index=False)))
    assert ds['cell']['eol_cycle']==2
    with pytest.raises(ValueError): load_sensor_csv(io.StringIO('x\n1\n'))

def test_nasa_pairing_and_unpaired_eol():
    t=trace()
    charge={'type':'charge','data':{'Time':t.time.values,'Voltage_measured':t.voltage.values,
            'Current_measured':t.current.values,'Temperature_measured':t.temperature.values}}
    discharge=lambda cap:{'type':'discharge','data':{'Capacity':cap}}
    # First unpaired discharge still defines initial reference. Final unpaired
    # discharge defines EOL but must never reuse the preceding charge input.
    records=np.array([discharge(2),charge,discharge(1.9),discharge(1.5)],dtype=object)
    buf=io.BytesIO();savemat(buf,{'B0005':{'cycle':records}});buf.seek(0)
    d=load_nasa_mat(buf)['B0005']
    assert d['specs']['c0']==2
    assert d['eol_cycle']==3
    assert d['summary'].cycle.tolist()==[2]
    assert d['summary'].rul.tolist()==[1]

def test_single_battery_rejected():
    with pytest.raises(ValueError): evaluate_lobo({'a':tiny()['a']})
