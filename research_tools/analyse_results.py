from pathlib import Path
import sys,json,platform,time
import numpy as np,pandas as pd,scipy,sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'python_app'))
from data_loader import load_battery_data
from models import build_features,fit_lobo_models,evaluate_lobo,quality_boundary,score
from simulator import DataQualitySimulator
D=load_battery_data(source='nasa',data_dir=R/'python_app/data/raw');F=build_features(D)
out=R/'results';assets=R/'assets';assets.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['Nimbus Roman','DejaVu Serif'],'font.size':12})
B=json.loads((out/'benchmark_summary.json').read_text());S=pd.DataFrame(B['sweeps'])
labels={'rf':'Random Forest','ridge':'Ridge','gb':'Gradient Boosting','rf_aug':'Augmented RF'}
colors={'rf':'#146f86','ridge':'#ad5d20','gb':'#47723e','rf_aug':'#795292'}
# Clean dataset exports and missingness are useful for independent review.
pd.concat(F.values()).to_csv(out/'clean_features.csv',index=False)
miss=pd.concat([x.assign(battery_id=b) for b,x in F.items()]).groupby('battery_id').apply(lambda x:x.isna().mean(),include_groups=False)
miss.to_csv(out/'feature_missing_fraction.csv')
pd.DataFrame(B['clean_benchmarks']).T.to_csv(out/'model_comparison.csv',index_label='model')
# Retain four independent cells when averaging pre-EOL errors.
P=pd.read_csv(out/'clean_predictions.csv');pre=[]
for model,part in P.groupby('model'):
 for bid,x in part.groupby('battery_id'):
  z=x[x.rul>0];a,b=score(z.rul,z.pred_rul)
  pre.append({'model':model,'battery_id':bid,'n_pre_eol':len(z),'rul_mae':a,'rul_rmse':b})
pd.DataFrame(pre).to_csv(out/'pre_eol_rul_folds.csv',index=False)
premean=pd.DataFrame(pre).groupby('model')[['rul_mae','rul_rmse']].mean()
premean.to_csv(out/'pre_eol_rul_summary.csv')
# Supplementary scenarios do not replace the predeclared single-factor grids.
extras=[];fitted=fit_lobo_models(D,'rf',clean=F)
for name,sim in [('Combined stress',DataQualitySimulator(sampling_step=10,noise_std_voltage=.05,noise_std_current=.1,noise_std_temp=1.25,missing_rate=.5,charge_window_pct=.5)),('Upper 50% charge',DataQualitySimulator(charge_window_pct=.5,charge_window_start=.5))]:
 for seed in [42,43,44]:
  res=evaluate_lobo(D,fitted=fitted,clean=F,test_simulator=sim.clone(seed))
  extras.append({'scenario':name,'repeat_seed':seed,**{m:res[m] for m in ['soh_mae','soh_rmse','rul_mae','rul_rmse']}})
no_cycle=fit_lobo_models(D,'rf',feature_cols=[c for c in F['B0005'].columns if c in fitted['B0005'].feature_cols and c!='cycle'],clean=F)
res=evaluate_lobo(D,fitted=no_cycle,clean=F)
extras.append({'scenario':'RF without cycle feature','repeat_seed':None,**{m:res[m] for m in ['soh_mae','soh_rmse','rul_mae','rul_rmse']}})
pd.DataFrame(extras).to_csv(out/'supplementary_experiments.csv',index=False)
bounds=[]
for model in labels:
 for factor in S.factor.unique():
  for metric,tol in [('soh_mae',3),('rul_mae',15)]:
   q=quality_boundary(S[(S.model==model)&(S.factor==factor)],tol,B['clean_benchmarks']['Cycle Count Baseline'][metric],metric,reverse=factor=='charge_window_pct')
   bounds.append({'model':model,'factor':factor,'metric':metric,'illustrative_tolerance':tol,**q})
pd.DataFrame(bounds).to_csv(out/'quality_boundaries.csv',index=False)
# Scientifically labelled plots from recorded predictions.
fig,ax=plt.subplots(figsize=(7.3,4.6))
for bid,d in D.items():ax.plot(d['summary'].cycle,d['summary'].soh,label=bid,lw=1.6)
ax.axhline(80,color='#666666',ls=':',label='Project EOL 80%');ax.set(xlabel='Discharge cycle index',ylabel='Observed SOH (%)');ax.legend(ncol=3,frameon=False);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(assets/'capacity_trajectories.png',dpi=230);plt.close(fig)
for metric in ['soh_mae','rul_mae']:
 for group in range(2):
  factors=list(S.factor.unique())[group*2:group*2+2]
  fig,axs=plt.subplots(2,1,figsize=(7.2,7.7));fig.subplots_adjust(left=.14,right=.98,top=.96,bottom=.23,hspace=.52)
  for ax,factor in zip(axs,factors):
   mul=1000 if factor=='noise_std_voltage' else 100 if factor in ['missing_rate','charge_window_pct'] else 1
   for key in labels:
    z=S[(S.model==key)&(S.factor==factor)].groupby('level')[metric].mean().sort_index(ascending=factor!='charge_window_pct')
    ax.plot(z.index*mul,z.values,marker='o',markersize=4,label=labels[key],color=colors[key],lw=1.5)
   ax.axhline(3 if metric=='soh_mae' else 15,color='#555555',ls=':',label='Illustrative tolerance')
   ax.axhline(B['clean_benchmarks']['Cycle Count Baseline'][metric],color='#888888',ls='--',label='Cycle-only baseline')
   ax.set_xlabel({'sampling_step':'Keep every Nth reading','noise_std_voltage':'Voltage noise standard deviation (mV)','missing_rate':'Dropped readings (%)','charge_window_pct':'Early charge duration retained (%)'}[factor])
   ax.set_ylabel('SOH MAE (pp)' if metric=='soh_mae' else 'RUL MAE (cycles)');ax.grid(axis='y',alpha=.25);ax.spines[['top','right']].set_visible(False)
   if factor=='charge_window_pct':ax.invert_xaxis()
  fig.legend(*axs[0].get_legend_handles_labels(),ncol=2,loc='lower center',frameon=False,fontsize=11)
  fig.savefig(assets/f'{metric}_{group+1}.png',dpi=230);plt.close(fig)
fig,axs=plt.subplots(2,1,figsize=(7.2,6.8),sharex=True)
for model,c in zip(B['clean_benchmarks'],['#777777','#ad5d20','#146f86','#47723e']):
 x=P[(P.model==model)&(P.battery_id=='B0005')];axs[0].plot(x.cycle,x.pred_soh,label=model,color=c,lw=1.3);axs[1].plot(x.cycle,x.pred_rul,label=model,color=c,lw=1.3)
x=P[(P.model=='Random Forest')&(P.battery_id=='B0005')]
axs[0].plot(x.cycle,x.soh,'k--',label='Observed',lw=1.7);axs[1].plot(x.cycle,x.rul,'k--',label='Observed',lw=1.7)
axs[0].set_ylabel('SOH (%)');axs[1].set_ylabel('RUL (cycles)');axs[1].set_xlabel('Cycle index');axs[0].legend(fontsize=9,ncol=2,frameon=False);fig.tight_layout();fig.savefig(assets/'heldout_predictions.png',dpi=230);plt.close(fig)
I=pd.DataFrame(B['permutation_importance']).groupby('feature').soh_mae_increase.mean().sort_values()
fig,ax=plt.subplots(figsize=(7.2,4.4));ax.barh(I.index,I.values,color='#146f86');ax.set_xlabel('Held-out SOH MAE increase after permutation (pp)');fig.tight_layout();fig.savefig(assets/'importance.png',dpi=230);plt.close(fig)
E={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__,'platform':platform.platform(),'processor':platform.processor()}
(out/'environment.json').write_text(json.dumps(E,indent=2))
summary={'clean':B['clean_benchmarks'],'pre_eol':premean.to_dict('index'),'supplementary':extras,'rf_boundaries':[x for x in bounds if x['model']=='rf'],'missing_fraction':miss.round(4).to_dict('index')}
(out/'analysis_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False));print(json.dumps(summary,indent=2),flush=True)
