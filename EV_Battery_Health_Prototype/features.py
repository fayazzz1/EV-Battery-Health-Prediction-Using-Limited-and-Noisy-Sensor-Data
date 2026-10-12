"""Charge features with honest missing values and fold-local imputation downstream."""
import numpy as np
import pandas as pd
from data_loader import stable_seed

FEATURE_GROUPS={'Voltage':['v_mean','v_std','v_min','v_max','v_skew'],
                'Timing':['charging_time_sec','v_trans_time_sec'],
                'Temperature':['t_max','t_rise'],'Charge':['coulomb_capacity_ah'],'Cycle':['cycle']}
FEATURE_NAMES=['cycle']+[c for g,cols in FEATURE_GROUPS.items() if g!='Cycle' for c in cols]

def extract_cycle_features(df_cycle,cycle_num):
    result={c:np.nan for c in FEATURE_NAMES};result['cycle']=cycle_num
    df=df_cycle.replace([np.inf,-np.inf],np.nan).dropna(subset=['time']).sort_values('time').drop_duplicates('time')
    if len(df)<3: return result
    v=df.voltage.dropna();temp=df.temperature.dropna()
    result['charging_time_sec']=float(df.time.iloc[-1]-df.time.iloc[0])
    if len(v)>=3:
        result.update(v_mean=float(v.mean()),v_std=float(v.std(ddof=0)),v_min=float(v.min()),v_max=float(v.max()),v_skew=float(v.skew()) if v.std()>1e-12 else 0.)
    if len(temp)>=2: result.update(t_max=float(temp.max()),t_rise=float(max(0,temp.max()-temp.iloc[0])))
    pairs=df[['time','current']].dropna()
    if len(pairs)>=2:
        dt=np.diff(pairs.time.to_numpy());i=np.maximum(pairs.current.to_numpy(),0)
        usable=dt<=5*np.median(dt)
        result['coulomb_capacity_ah']=float(np.sum(.5*(i[:-1]+i[1:])*dt*usable)/3600)
    valid=df[['time','voltage']].dropna()
    if len(valid)>1 and valid.voltage.iloc[0]<=3.8:
        t1=valid.loc[valid.voltage>=3.8,'time'];t2=valid.loc[valid.voltage>=4.1,'time']
        if len(t1) and len(t2): result['v_trans_time_sec']=float(max(0,t2.iloc[0]-t1.iloc[0]))
    return result

def extract_features_for_battery(battery_data,simulator=None):
    rows=[]
    for _,row in battery_data['summary'].iterrows():
        k=int(row.cycle);trace=battery_data['timeseries'][k]
        if simulator is not None:
            trace=simulator.clone(stable_seed(simulator.seed,row.battery_id,k)).apply(trace)
        record=extract_cycle_features(trace,k)
        for c in ['battery_id','capacity_ah','soh','rul']: record[c]=row[c]
        rows.append(record)
    return pd.DataFrame(rows)
