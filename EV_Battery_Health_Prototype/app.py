"""Streamlit research dashboard for battery-disjoint data-quality experiments."""
from pathlib import Path
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from data_loader import load_battery_data,load_nasa_mat,load_sensor_csv,dataset_fingerprint
from simulator import DataQualitySimulator
from features import FEATURE_NAMES,extract_cycle_features,extract_features_for_battery
from models import ML_MODEL_REGISTRY,fit_lobo_models,quality_boundary
from precompute_benchmarks import run_all_benchmarks,code_fingerprint,SCHEMA_VERSION

st.set_page_config(page_title='Battery Health Lab',page_icon='⚡',layout='wide')
st.markdown('''<style>
.stApp {background:#f6f8fc;} h1,h2,h3 {color:#12243a;}
[data-testid="stMetric"] {background:white;border:1px solid #e0e7ee;border-radius:14px;padding:18px;}
[data-testid="stSidebar"] {background:#eaf0f6;}
</style>''',unsafe_allow_html=True)
st.caption('SPECIAL PROJECT  ·  DATA QUALITY & BATTERY HEALTH')
st.title('⚡ Battery Health Lab')
st.write('Explore how incomplete and noisy charge measurements affect predictions on an unseen battery.')

@st.cache_data(show_spinner=False)
def synthetic(): return load_battery_data()

@st.cache_data(show_spinner=False)
def imported(files,source):
    ds={}
    for name,content in files:
        part=load_nasa_mat(io.BytesIO(content)) if source=='NASA MAT files' else load_sensor_csv(io.BytesIO(content))
        if set(ds)&set(part): raise ValueError('Duplicate battery IDs in uploaded files.')
        ds.update(part)
    return ds

st.sidebar.title('Experiment controls')
source=st.sidebar.selectbox('Data source',['Synthetic demonstration','NASA MAT files','Normalized sensor CSV'])
if source=='Synthetic demonstration':
    dataset=synthetic()
    st.warning('Synthetic demonstration: generated signals encode simulated ageing. Accuracy here is a software check, not evidence of NASA or real-EV performance.')
else:
    uploads=st.sidebar.file_uploader('Upload battery records',type=['mat'] if source=='NASA MAT files' else ['csv'],accept_multiple_files=True)
    if not uploads:
        st.info('Upload at least two independent batteries. CSV columns: battery_id, cycle, time, voltage, current, temperature, capacity_ah. Units: seconds, V, A, °C, Ah. capacity_ah is a discharge label; sensor rows describe the matching charge.')
        st.stop()
    try: dataset=imported(tuple((u.name,u.getvalue()) for u in uploads),source)
    except (ValueError,KeyError,TypeError,NotImplementedError,OSError) as exc:
        st.error('Could not import data: '+str(exc));st.stop()
    st.info('Imported records. Verify their provenance before treating results as laboratory evidence; evaluation is descriptive.')
if len(dataset)<2:
    st.error('LOBO requires at least two independent batteries.');st.stop()
fingerprint=dataset_fingerprint(dataset)
battery=st.sidebar.selectbox('Held-out battery',list(dataset))
cycles=dataset[battery]['summary'].cycle.astype(int).tolist()
cycle=st.sidebar.select_slider('Cycle',options=cycles,value=cycles[len(cycles)//2])
model_label=st.sidebar.selectbox('Prediction model',list(ML_MODEL_REGISTRY),index=2)
augmented=st.sidebar.checkbox('Train with degraded data',False)
st.sidebar.divider()
st.sidebar.subheader('Sensor quality')
step=st.sidebar.select_slider('Keep every Nth reading',options=[1,2,5,10,20,30],value=1)
noise=st.sidebar.slider('Voltage noise σ (mV)',0,100,0,5)
st.sidebar.caption('Linked current σ = 2 × voltage σ (A/V); temperature σ = 25 × voltage σ (°C/V).')
precision=st.sidebar.select_slider('Voltage rounding (mV)',options=[0,1,5,10,20,50],value=0)
current_precision=st.sidebar.select_slider('Current rounding (A)',options=[0.,.01,.05,.1],value=0.)
temp_precision=st.sidebar.select_slider('Temperature rounding (°C)',options=[0.,.1,.5,1.],value=0.)
missing=st.sidebar.slider('Random missing readings (%)',0,90,0,5)
burst=st.sidebar.checkbox('Add a 15% consecutive gap',False)
window=st.sidebar.slider('Retained charge duration (%)',10,100,100,5)
window_location=st.sidebar.radio('Charge segment',['Early','Upper'],horizontal=True)
seed=st.sidebar.number_input('Sensor seed',0,999999,42)
sim=DataQualitySimulator(sampling_step=step,noise_std_voltage=noise/1000,
    noise_std_current=noise/500,noise_std_temp=noise/40,voltage_precision=precision/1000 if precision else None,
    current_precision=current_precision or None,temp_precision=temp_precision or None,
    missing_rate=missing/100,burst_drop_gap=burst,charge_window_pct=window/100,
    charge_window_start=0 if window_location=='Early' else 1-window/100,seed=int(seed))

@st.cache_resource(show_spinner='Training on the other batteries…')
def train(fp,key,augmentation,_dataset):
    return fit_lobo_models(_dataset,key,augmentation)

fit=train(fingerprint,ML_MODEL_REGISTRY[model_label],augmented,dataset)[battery]
raw=dataset[battery]['timeseries'][cycle]
degraded=sim.apply(raw)
features=pd.DataFrame([extract_cycle_features(degraded,cycle)])
ps,pr=fit.predict(features)
actual=dataset[battery]['summary'].set_index('cycle').loc[cycle]
tab_live,tab_robust,tab_validation,tab_features,tab_method=st.tabs(['Live prediction','Robustness experiments','Cross-battery results','Features & ablation','Data & methodology'])

def number(value,digits=1): return f'{value:.{digits}f}' if pd.notna(value) else 'Unknown'

with tab_live:
    st.subheader(f'{battery} · cycle {cycle}')
    st.caption('Training batteries: '+', '.join(b for b in dataset if b!=battery))
    a,b,c,d=st.columns(4)
    a.metric('Predicted SOH',number(ps[0])+'%',f'{ps[0]-actual.soh:+.2f} pp vs observed',delta_color='off')
    b.metric('Predicted RUL',number(pr[0],0)+' cycles')
    c.metric('Health grade','Good' if ps[0]>=90 else 'Moderate' if ps[0]>=80 else 'Below 80%')
    d.metric('Readings retained',f'{len(degraded)/len(raw):.1%}',f'{len(degraded):,} / {len(raw):,}',delta_color='off')
    st.caption(f'Observed SOH: {actual.soh:.2f}% · Observed RUL: {number(actual.rul,0)} cycles. Grades are project-defined; RUL is unknown for cells without observed EOL.')
    if features[FEATURE_NAMES].isna().any().any():
        st.warning('Some features are unavailable in this segment. They are imputed using training-battery values. A point prediction alone does not prove reliability.')
    if ps[0]<0 or ps[0]>110: st.warning('SOH prediction is outside a plausible range; inspect the input and model extrapolation.')
    signal=st.radio('Sensor signal',['voltage','current','temperature'],horizontal=True)
    chart=go.Figure()
    chart.add_trace(go.Scatter(x=raw.time,y=raw[signal],name='Original',line={'color':'#9bacbf'}))
    chart.add_trace(go.Scatter(x=degraded.time,y=degraded[signal],name='Degraded',line={'color':'#137b89'}))
    units={'voltage':'V','current':'A','temperature':'°C'}
    chart.update_layout(xaxis_title='Time (seconds)',yaxis_title=f'{signal.title()} ({units[signal]})',height=350)
    st.plotly_chart(chart,width='stretch')
    trajectory=dataset[battery]['summary']
    chart=px.line(trajectory,x='cycle',y='soh',title='Observed capacity trajectory')
    chart.add_hline(y=80,line_dash='dash',annotation_text='Project EOL: 80%')
    chart.add_trace(go.Scatter(x=[cycle],y=[ps[0]],mode='markers',name='Current prediction',marker={'size':12}))
    st.plotly_chart(chart,width='stretch')
    download=features.assign(battery_id=battery,predicted_soh=ps[0],predicted_rul=pr[0],observed_soh=actual.soh,observed_rul=actual.rul)
    st.download_button('Download this prediction',download.to_csv(index=False),'prediction.csv','text/csv')

@st.cache_data(show_spinner=False)
def saved_benchmark(modified):
    path=Path(__file__).parent/'data'/'benchmark_summary.json'
    if not path.exists(): return None
    try: return json.loads(path.read_text())
    except (ValueError,OSError): return None

benchmark_path=Path(__file__).parent/'data'/'benchmark_summary.json'
saved=saved_benchmark(benchmark_path.stat().st_mtime_ns if benchmark_path.exists() else 0)
if saved and (saved.get('metadata',{}).get('dataset_fingerprint')!=fingerprint or
              saved.get('metadata',{}).get('code_fingerprint')!=code_fingerprint() or
              saved.get('metadata',{}).get('schema_version')!=SCHEMA_VERSION): saved=None
session_key='experiment_'+fingerprint+'_'+code_fingerprint()
results=st.session_state.get(session_key,saved)
with tab_robust:
    st.subheader('Measure the limits of sensor quality')
    st.write('Each factor changes on its own. Models train on other batteries; repeated seeds measure sensor randomness. Augmented RF trains on clean plus degraded versions of training batteries.')
    a,b,c=st.columns(3)
    tolerance=a.number_input('SOH tolerance (percentage points)',.1,30.,3.,.1)
    rul_tolerance=b.number_input('RUL tolerance (cycles)',1.,200.,15.,1.)
    repeats=c.selectbox('Repeated sensor seeds',[1,3,5],index=1)
    st.caption('Choose and justify tolerances before running experiments. They are project choices, not safety guarantees.')
    if st.button('Run all experiments',type='primary'):
        with st.spinner('Running LOBO, four factor sweeps, degraded training and feature ablations…'):
            results=run_all_benchmarks(dataset,repeats=repeats)
            st.session_state[session_key]=results
    if results:
        st.caption('Loaded results: '+str(results['metadata']['repeats'])+' sensor seeds; '+('quick endpoint grid' if results['metadata']['quick_grid'] else 'full grid')+'.')
        sweeps=pd.DataFrame(results['sweeps'])
        chosen=st.selectbox('Compare model',['rf','ridge','gb','rf_aug'])
        metric=st.selectbox('Error metric',['soh_mae','soh_rmse','rul_mae','rul_rmse'])
        baseline=results['clean_benchmarks']['Cycle Count Baseline'][metric]
        tol=tolerance if metric.startswith('soh') else rul_tolerance
        rows=[]
        for factor in sweeps.factor.unique():
            data=sweeps[(sweeps.factor==factor)&(sweeps.model==chosen)]
            summary=data.groupby('level',as_index=False)[metric].agg(['mean','std']).reset_index()
            chart=px.line(summary,x='level',y='mean',error_y='std',markers=True,title=factor.replace('_',' ').title())
            chart.update_layout(yaxis_title=metric+' (battery mean)',xaxis_title='Keep every Nth reading' if factor=='sampling_step' else 'Factor level')
            chart.add_hline(y=tol,line_dash='dot',annotation_text='Chosen tolerance')
            if baseline is not None: chart.add_hline(y=baseline,line_dash='dash',annotation_text='Cycle-only baseline')
            if factor=='charge_window_pct': chart.update_xaxes(autorange='reversed')
            st.plotly_chart(chart,width='stretch')
            bound=quality_boundary(data,tol,baseline if baseline is not None else np.inf,metric,reverse=factor=='charge_window_pct')
            rows.append({'factor':factor,**bound})
        st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
        st.caption('Limits refer only to sampled levels, ordered from clean to severe. Blank failure means no crossing in the tested grid; blank passing level means even the clean point fails. Error bars show seed SD, not confidence intervals.')
        st.download_button('Download experiment JSON',json.dumps(results,indent=2,allow_nan=False),'battery_experiments.json','application/json')
    else: st.info('Run experiments to calculate error curves and quality limits. No accuracy or limits are assumed.')

with tab_validation:
    st.subheader('One battery held out in every fold')
    st.caption(f'{len(dataset)} independent batteries. Primary metrics give equal weight to each battery; SOH errors use percentage points and RUL errors use cycles.')
    if results:
        st.dataframe(pd.DataFrame(results['clean_benchmarks']).T,width='stretch')
        st.dataframe(pd.DataFrame(results['folds']),hide_index=True,width='stretch')
        st.download_button('Download fold results',pd.DataFrame(results['folds']).to_csv(index=False),'lobo_folds.csv','text/csv')
    else: st.info('Use Run all experiments to evaluate the models.')
    st.write('Cells whose EOL is unobserved have no exact RUL target. Their SOH is still evaluated; direct RUL metrics exclude them. No test-battery labels are used for fitting or imputation.')

with tab_features:
    st.subheader('What information reaches the model?')
    st.dataframe(features.T.rename(columns={0:'Current value'}),width='stretch')
    st.caption('Integrated charge is observed charge over valid time intervals, not the discharge-capacity target. Gaps over five median sample intervals are excluded; shorter gaps use trapezoidal interpolation.')
    if results:
        importance=pd.DataFrame(results['permutation_importance']).groupby('feature',as_index=False).soh_mae_increase.mean().sort_values('soh_mae_increase')
        st.plotly_chart(px.bar(importance,x='soh_mae_increase',y='feature',orientation='h',title='Held-out permutation importance (SOH error increase)'),width='stretch')
        st.caption('Permutations occur only on held-out batteries. Correlated features can hide importance; this is a descriptive diagnostic.')
        st.subheader('Remove one feature group')
        st.dataframe(pd.DataFrame(results['ablations']),hide_index=True,width='stretch')
        st.caption('Each row retrains RF without the named group and tests on clean or 50% early-charge inputs. Compare with the full-feature RF clean/sweep result.')

with tab_method:
    st.subheader('Dataset audit')
    audit=[]
    for bid,data in dataset.items():
        summary=data['summary']; intervals=[trace.time.diff().median() for trace in data['timeseries'].values()]
        audit.append({'battery':bid,'source':data['source'],'paired_cycles':len(summary),
                      'initial_capacity_Ah':data['specs']['c0'],'EOL_cycle':data['eol_cycle'],
                      'RUL_observed':data['eol_cycle'] is not None,'median_sample_interval_s':float(np.median(intervals))})
    st.dataframe(pd.DataFrame(audit),hide_index=True,width='stretch')
    st.latex(r'SOH_k = 100\, C_k/C_0,\qquad RUL_k=\max(0,k_{EOL}-k)')
    st.write('C₀ is the first measured discharge capacity. EOL is the first observed cycle with SOH ≤ 80%. Labels come from the original capacity trajectory and never change when sensors are degraded.')
    st.write('Pipeline: import → charge/discharge pairing → fixed labels → sensor degradation → charge features → fold-local imputation and scaling → SOH/RUL models → battery-disjoint evaluation → robustness limits.')
    st.write('Sampling is decimation by reading index, not a fixed number of seconds. Noise, rounding, random loss, burst loss and early/upper charge windows have separate live controls. Published sweeps isolate the four core factors; the noise sweep does not add rounding or bursts.')
    st.write('The proposal, literature review and design report motivate the four models, cross-battery validation, controlled degradation, augmentation, and feature comparisons. NASA is supported through original MAT files; CALCE needs conversion to the normalized CSV schema. No real dataset was included with the supplied prototype.')
    st.caption('Lab cells and simulated telemetry loss do not establish field performance. Few independent batteries limit generalization. Thresholds and health grades are research choices.')
