"""Explicit synthetic demo and real NASA MAT / normalized CSV loaders."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from scipy.io import loadmat

DATA_DIR = Path(__file__).parent / 'data'
BATTERY_SPECS = {'B0005': (168, 1.86, .0038), 'B0006': (168, 2.04, .0054),
                 'B0007': (168, 1.89, .0032), 'B0018': (132, 1.86, .0052)}

def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:4], 'little')

def label_battery(summary, timeseries, source, threshold=80.):
    summary = summary.sort_values('cycle').reset_index(drop=True).copy()
    if summary.empty or summary.cycle.duplicated().any():
        raise ValueError('Each battery needs unique, nonempty capacity cycles.')
    if not np.isfinite(summary.capacity_ah).all() or (summary.capacity_ah<=0).any():
        raise ValueError('Capacities must be finite and positive.')
    c0 = float(summary.capacity_ah.iloc[0])
    summary['initial_capacity_ah'] = c0
    summary['soh'] = 100 * summary.capacity_ah / c0
    crossed = summary.loc[summary.soh<=threshold,'cycle']
    eol = int(crossed.iloc[0]) if len(crossed) else None
    summary['rul'] = np.maximum(0,eol-summary.cycle) if eol is not None else np.nan
    summary['rul_observed'] = eol is not None
    return {'summary':summary, 'timeseries':timeseries, 'source':source,
            'eol_cycle':eol,'threshold':threshold,'specs':{'c0':c0,'cycles':len(summary)}}

def generate_cycle_timeseries(battery_id, cycle_num, capacity_ah, initial_capacity, seed=42):
    rng = np.random.default_rng(stable_seed(seed,battery_id,cycle_num,'sensor'))
    ratio = capacity_ah/initial_capacity
    cc, cv = int(2200*ratio), int(1800*ratio+100)
    time = np.arange(cc+cv,dtype=float)
    u = np.linspace(0,1,cc)
    v0 = 3.6+.15*(1-ratio)
    voltage = np.r_[v0+(4.2-v0)*(1-np.exp(-3.5*u))/(1-np.exp(-3.5)),np.full(cv,4.2)]
    current = np.r_[np.full(cc,1.5),1.5*np.exp(np.log(.02/1.5)*np.linspace(0,1,cv))]
    temp = 24+rng.normal(0,.2)+(3.5+4*(1-ratio))*np.sin(np.pi*np.clip(time/time[-1]*1.4,0,1))**1.2
    return pd.DataFrame({'time':time,'voltage':voltage+rng.normal(0,.001,len(time)),
                         'current':current+rng.normal(0,.002,len(time)),
                         'temperature':temp+rng.normal(0,.02,len(time))})

def generate_nasa_dataset(seed=42, threshold=80.):
    """Legacy name; produces SYNTHETIC data, never NASA observations."""
    ds={}
    for bid,(n,c0,rate) in BATTERY_SPECS.items():
        rng=np.random.default_rng(stable_seed(seed,bid,'capacity'))
        rows,traces,capacity=[],{},c0
        for k in range(1,n+1):
            if k>1:
                capacity-=rate*(1+.3*np.sqrt(k/50))+rng.normal(0,.0012)
                if k%35==0: capacity+=.025
            rows.append({'battery_id':bid,'cycle':k,'capacity_ah':capacity})
            traces[k]=generate_cycle_timeseries(bid,k,capacity,c0,seed)
        ds[bid]=label_battery(pd.DataFrame(rows),traces,'Synthetic demonstration',threshold)
    return ds

def load_nasa_mat(path_or_buffer, threshold=80.):
    """Original NASA v5 structs. Pair preceding charge with discharge label.

    All observed capacities define initial capacity and EOL, even if some
    discharges have no usable charge. Only paired cycles become model inputs.
    """
    raw=loadmat(path_or_buffer,simplify_cells=True)
    ds={}
    for bid,cell in raw.items():
        if bid.startswith('__') or not isinstance(cell,dict) or 'cycle' not in cell: continue
        records=cell['cycle']
        if isinstance(records,dict): records=[records]
        rows,traces,pending,k=[],{},None,0
        for rec in records:
            data=rec['data']; kind=str(rec['type']).strip()
            if kind=='charge':
                cols={'time':'Time','voltage':'Voltage_measured','current':'Current_measured','temperature':'Temperature_measured'}
                pending=pd.DataFrame({out:np.atleast_1d(data[key]).astype(float).ravel() for out,key in cols.items()})
                pending=pending.replace([np.inf,-np.inf],np.nan).dropna(subset=['time']).sort_values('time').drop_duplicates('time')
            elif kind=='discharge':
                k+=1
                cap=float(np.asarray(data.get('Capacity',np.nan)).ravel()[0])
                if np.isfinite(cap) and cap>0:
                    rows.append({'battery_id':bid,'cycle':k,'capacity_ah':cap})
                    if pending is not None and len(pending)>=3: traces[k]=pending.reset_index(drop=True)
                pending=None
        if rows:
            labeled=label_battery(pd.DataFrame(rows),traces,'NASA MAT import',threshold)
            labeled['summary']=labeled['summary'][labeled['summary'].cycle.isin(traces)].reset_index(drop=True)
            if not labeled['summary'].empty: ds[bid]=labeled
    if not ds: raise ValueError('No usable charge/discharge pairs. Original NASA v5 MAT is required; v7.3 is unsupported.')
    return ds

def load_sensor_csv(path_or_buffer, threshold=80.):
    """Common CSV schema for CALCE/other data after source-specific conversion."""
    df=pd.read_csv(path_or_buffer)
    required=['battery_id','cycle','time','voltage','current','temperature','capacity_ah']
    missing=set(required)-set(df.columns)
    if missing: raise ValueError('Missing CSV columns: '+', '.join(sorted(missing)))
    if df.empty or df[required].isna().any().any(): raise ValueError('Required values cannot be missing.')
    numeric=required[1:]; df[numeric]=df[numeric].apply(pd.to_numeric,errors='raise')
    if not np.isfinite(df[numeric]).all().all() or (df.cycle<1).any() or (df.cycle%1!=0).any():
        raise ValueError('Finite numeric values and positive integer cycles required.')
    ds={}
    for bid,group in df.groupby('battery_id'):
        rows,traces=[],{}
        for k,trace in group.groupby('cycle'):
            if trace.capacity_ah.nunique()!=1: raise ValueError('One discharge capacity label per cycle required.')
            trace=trace.sort_values('time')
            if len(trace)<3 or trace.time.duplicated().any(): raise ValueError('At least 3 unique timestamps per cycle required.')
            rows.append({'battery_id':str(bid),'cycle':int(k),'capacity_ah':float(trace.capacity_ah.iloc[0])})
            traces[int(k)]=trace[['time','voltage','current','temperature']].reset_index(drop=True)
        ds[str(bid)]=label_battery(pd.DataFrame(rows),traces,'Normalized CSV import',threshold)
    return ds

def load_battery_data(force_reload=False,source='synthetic',data_dir=None,seed=42,threshold=80.):
    if source=='synthetic': return generate_nasa_dataset(seed,threshold)
    if source not in ('nasa','csv'): raise ValueError('Source must be synthetic, nasa or csv.')
    root=Path(data_dir or DATA_DIR/'raw'); files=sorted(root.glob('*.mat' if source=='nasa' else '*.csv'))
    if not files: raise ValueError('No matching files in '+str(root))
    ds={}
    for p in files:
        part=load_nasa_mat(p,threshold) if source=='nasa' else load_sensor_csv(p,threshold)
        if set(ds)&set(part): raise ValueError('Duplicate battery IDs across files.')
        ds.update(part)
    return ds

def dataset_fingerprint(dataset):
    h=hashlib.sha256()
    for bid in sorted(dataset):
        data=dataset[bid]; h.update(str((bid,data['source'],data['threshold'])).encode())
        h.update(pd.util.hash_pandas_object(data['summary'],index=False).values.tobytes())
        for k in sorted(data['timeseries']):
            h.update(str(k).encode());h.update(pd.util.hash_pandas_object(data['timeseries'][k],index=False).values.tobytes())
    return h.hexdigest()
