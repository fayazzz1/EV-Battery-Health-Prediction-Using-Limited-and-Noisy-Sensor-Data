import {decodeTrace,degrade,features,predict} from './engine.mjs?v=3';
const $=id=>document.getElementById(id);
const fmt=(x,d=2)=>Number.isFinite(x)?x.toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d}):'Unknown';
const names={rf:'Random Forest',ridge:'Ridge Regression',gb:'Gradient Boosting',cycle_only:'Cycle Count Baseline',rf_aug:'Augmented Random Forest'};
const cache=new Map();let manifest,bench,current,revision=0,timer;
const status=(text,error=false)=>{$('status').textContent=text;$('status').dataset.state=error?'error':'ready';};
async function get(path,binary=false){
  if(!cache.has(path))cache.set(path,fetch(path).then(r=>{if(!r.ok)throw Error(`Could not load ${path} (${r.status}).`);return binary?r.arrayBuffer():r.json();}).catch(e=>{cache.delete(path);throw e;}));
  return cache.get(path);
}
function config(){return {step:+$('step').value,noise:+$('noise').value/1000,missing:+$('missing').value/100,window:+$('window').value/100,upper:$('segment').value==='upper',rounding:+$('rounding').value/1000,burst:$('burst').checked,seed:+$('seed').value};}
function labels(){for(const [id,unit] of [['cycle',''],['noise',' mV'],['missing','%'],['window','%']])$(id+'-label').textContent=$(id).value+unit;}
function table(id,columns,rows){
  const t=document.createElement('table'),head=document.createElement('thead'),tr=document.createElement('tr');
  columns.forEach(c=>{const th=document.createElement('th');th.textContent=c;tr.append(th);});head.append(tr);t.append(head);
  const body=document.createElement('tbody');
  rows.forEach(values=>{const tr=document.createElement('tr');values.forEach(v=>{const td=document.createElement('td');td.textContent=typeof v==='number'?fmt(v):v??'Unknown';if(typeof v==='number')td.className='num';tr.append(td);});body.append(tr);});t.append(body);$(id).replaceChildren(t);
}
function plot(id,series,{xlabel='',ylabel='',references=[],reverse=false}={}){
  const root=$(id);if(!root.clientWidth)return;
  const width=Math.max(root.clientWidth,280),height=300,left=62,right=20,top=20,bottom=50;
  const NS='http://www.w3.org/2000/svg',svg=document.createElementNS(NS,'svg');svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
  const add=(tag,attrs,text)=>{const node=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs||{}))node.setAttribute(k,v);if(text!==undefined)node.textContent=text;svg.append(node);return node;};
  const points=series.flatMap(s=>s.points).filter(p=>p.every(Number.isFinite));if(!points.length){root.textContent='No values available.';return;}
  let xmin=Math.min(...points.map(p=>p[0])),xmax=Math.max(...points.map(p=>p[0]));if(xmin===xmax){xmin-=1;xmax+=1;}
  const ys=[...points.map(p=>p[1]),...references.map(r=>r.value)].filter(Number.isFinite);
  let ymin=Math.min(...ys),ymax=Math.max(...ys),pad=(ymax-ymin||1)*.1;ymin-=pad;ymax+=pad;
  const x=v=>left+(reverse?(xmax-v):(v-xmin))/(xmax-xmin)*(width-left-right);
  const y=v=>height-bottom-(v-ymin)/(ymax-ymin)*(height-top-bottom);
  const tick=v=>Math.abs(v)>=100?String(Math.round(v)):Number(v.toFixed(2)).toString();
  for(let i=0;i<=4;i++){const value=ymin+(ymax-ymin)*i/4,py=y(value);add('line',{x1:left,y1:py,x2:width-right,y2:py,stroke:'#e4ebf0'});add('text',{x:left-9,y:py+4,'text-anchor':'end'},tick(value));}
  const ticks=width<450?3:5;
  for(let i=0;i<=ticks;i++){const value=xmin+(xmax-xmin)*i/ticks;add('text',{x:x(value),y:height-bottom+20,'text-anchor':i===0?(reverse?'end':'start'):i===ticks?(reverse?'start':'end'):'middle'},tick(value));}
  add('text',{x:left+(width-left-right)/2,y:height-7,'text-anchor':'middle'},xlabel);
  add('text',{x:15,y:(height-bottom+top)/2,transform:`rotate(-90 15 ${(height-bottom+top)/2})`,'text-anchor':'middle'},ylabel);
  for(const r of references)if(Number.isFinite(r.value))add('line',{x1:left,x2:width-right,y1:y(r.value),y2:y(r.value),stroke:r.color,'stroke-dasharray':'5 4','stroke-width':1.5});
  for(const s of series){
    const valid=s.points.filter(p=>p.every(Number.isFinite)),every=Math.max(1,Math.floor(valid.length/250));
    const pts=valid.filter((_,i)=>i%every===0||i===valid.length-1);
    if(pts.length>1)add('path',{d:pts.map((p,i)=>`${i?'L':'M'}${x(p[0])},${y(p[1])}`).join(' '),fill:'none',stroke:s.color,'stroke-width':s.width||2});
    if(pts.length===1||s.markers)for(const p of pts){const dot=add('circle',{cx:x(p[0]),cy:y(p[1]),r:pts.length===1?6:4,fill:s.color});const title=document.createElementNS(NS,'title');title.textContent=`${s.name}: ${fmt(p[0])}, ${fmt(p[1])}`;dot.append(title);}
  }
  root.replaceChildren(svg);
}
function redrawLive(){
  if(!current)return;
  const {raw,degraded,truth,result,battery,cycle}=current,col=+$('signal').value;
  plot('signal-chart',[{name:'Original',color:'#94a9b9',points:raw.map(r=>[r[0],r[col]])},{name:'Degraded',color:'#087d8c',points:degraded.map(r=>[r[0],r[col]])}],{xlabel:'Time (seconds)',ylabel:['','Voltage (V)','Current (A)','Temperature (°C)'][col]});
  plot('trajectory-chart',[{name:'Observed SOH',color:'#087d8c',points:manifest.batteries[battery].summary.map(r=>[r.cycle,r.soh])},{name:'Prediction',color:'#ba620b',points:[[cycle,result.soh]]}],{xlabel:'Cycle number',ylabel:'SOH (%)',references:[{value:80,color:'#ba620b'}]});
  $('truth').textContent=`Ground truth: SOH ${fmt(truth.soh)}% · RUL ${fmt(truth.rul,0)} cycles · Dashed line: 80% end-of-life threshold.`;
}
async function update(){
  if(!manifest)return;const version=++revision;
  labels();const battery=$('battery').value;
  const summary=manifest.batteries[battery].summary;
  $('cycle').min=summary[0].cycle;$('cycle').max=summary.at(-1).cycle;
  if(!summary.some(r=>r.cycle===+$('cycle').value))$('cycle').value=summary.reduce((best,r)=>Math.abs(r.cycle-+$('cycle').value)<Math.abs(best.cycle-+$('cycle').value)?r:best).cycle;labels();
  const cycle=+$('cycle').value,modelKey=$('model').value,training=$('training').value,cfg=config();
  status('Calculating prediction from the selected measurements…');
  try{
    const [buffer,model]=await Promise.all([get(`./data/raw/${battery}-${cycle}.bin`,true),get(`./data/models/${battery}-${modelKey}-${training}.json`)]);
    if(version!==revision)return;
    const raw=decodeTrace(buffer),degraded=degrade(raw,cfg),f=features(degraded,cycle),result=predict(model,f),truth=summary.find(r=>r.cycle===cycle);
    current={battery,cycle,modelKey,training,cfg,raw,degraded,f,result,truth};
    $('selection').textContent=`${battery} · Cycle ${cycle}`;
    $('training-label').textContent='Trained on '+model.trained_on.join(', ');
    $('soh-value').textContent=fmt(result.soh)+'%';$('soh-detail').textContent=`Observed ${fmt(truth.soh)}% · error ${fmt(Math.abs(result.soh-truth.soh))} pp`;
    $('rul-value').textContent=fmt(result.rul,1);$('rul-detail').textContent=`cycles · observed ${fmt(truth.rul,0)} · error ${fmt(Math.abs(result.rul-truth.rul),1)}`;
    $('grade-value').textContent=result.soh>=90?'Good':result.soh>=80?'Moderate':'Below 80%';
    $('retained-value').textContent=fmt(100*degraded.length/raw.length,1)+'%';$('retained-detail').textContent=`${degraded.length.toLocaleString()} / ${raw.length.toLocaleString()} readings`;
    const warnings=[];if(result.soh<0||result.soh>110)warnings.push('SOH is outside a plausible range: this model is extrapolating poorly.');
    if(Object.values(f).some(v=>!Number.isFinite(v)))warnings.push('Unavailable features are imputed from training-battery values.');
    $('prediction-warning').textContent=warnings.join(' ');$('prediction-warning').hidden=!warnings.length;
    table('current-features',['Feature','Current value'],Object.entries(f).map(([k,v])=>[k,Number.isFinite(v)?v:'Unavailable → training imputation']));
    redrawLive();$('download-result').disabled=false;
    status(`Ready · ${names[modelKey]} · ${training==='aug'?'clean + degraded training':'clean training'} · sensor seed ${cfg.seed}`);
  }catch(e){if(version===revision)status(e.message+' Refresh to retry.',true);}
}
function refreshSweep(){
  if(!bench)return;
  const factor=$('factor').value,key=$('sweep-model').value,metric=$('metric').value,tolerance=+$('tolerance').value;
  const rows=bench.sweeps.filter(r=>r.factor===factor&&r.model===key),groups=new Map();
  for(const row of rows){if(!groups.has(row.level))groups.set(row.level,[]);groups.get(row.level).push(row[metric]);}
  const levels=[...groups.entries()].map(([level,values])=>({level,mean:values.reduce((a,b)=>a+b,0)/values.length,sd:Math.sqrt(values.reduce((a,b)=>a+(b-values.reduce((x,y)=>x+y,0)/values.length)**2,0)/Math.max(1,values.length-1))})).sort((a,b)=>factor==='charge_window_pct'?b.level-a.level:a.level-b.level);
  const baseline=bench.clean_benchmarks['Cycle Count Baseline'][metric];
  const multiplier=factor==='noise_std_voltage'?1000:['missing_rate','charge_window_pct'].includes(factor)?100:1;
  const xlabel={sampling_step:'Keep every Nth reading',noise_std_voltage:'Voltage noise σ (mV)',missing_rate:'Missing readings (%)',charge_window_pct:'Charge retained (%)'}[factor];
  plot('robustness-chart',[{name:names[key],color:'#087d8c',markers:true,points:levels.map(r=>[r.level*multiplier,r.mean])}],{xlabel,ylabel:metric.startsWith('soh')?'SOH error (pp)':'RUL error (cycles)',reverse:factor==='charge_window_pct',references:[{value:baseline,color:'#b45309'},{value:tolerance,color:'#8c6bad'}]});
  let last=null,failed=null,breakEven=null;for(const r of levels){if(failed===null){if(r.mean>tolerance)failed=r.level;else last=r.level;}if(breakEven===null&&r.mean>=baseline)breakEven=r.level;}
  const value=x=>x===null?'not observed':fmt(x*multiplier,2);
  $('boundary').textContent=`Last passing tested level: ${last===null?'none':value(last)} · First tolerance failure: ${value(failed)} · First level not beating baseline: ${value(breakEven)}`;
  table('sweep-table',[xlabel,'Mean error','Seed standard deviation'],levels.map(r=>[r.level*multiplier,r.mean,r.sd]));
}
function fillExperiments(){
  table('clean-results',['Model','SOH MAE (pp)','SOH RMSE (pp)','RUL MAE (cycles)','RUL RMSE (cycles)'],Object.entries(bench.clean_benchmarks).map(([name,r])=>[name,r.soh_mae,r.soh_rmse,r.rul_mae,r.rul_rmse]));
  table('fold-results',['Model','Test battery','Training batteries','SOH MAE (pp)','RUL MAE (cycles)'],bench.folds.map(r=>[r.model,r.test_battery,r.train_batteries,r.soh_mae,r.rul_mae]));
  const groups={};for(const r of bench.permutation_importance)(groups[r.feature]??=[]).push(r.soh_mae_increase);
  table('importance',['Feature','Mean SOH MAE increase (pp)'],Object.entries(groups).map(([n,v])=>[n,v.reduce((a,b)=>a+b,0)/v.length]).sort((a,b)=>b[1]-a[1]));
  table('ablations',['Removed group','Test condition','SOH MAE (pp)','RUL MAE (cycles)'],bench.ablations.map(r=>[r.removed_group,r.condition,r.soh_mae,r.rul_mae]));
  table('audit',['Cell','Paired cycles','Initial capacity (Ah)','Observed EOL cycle','Source'],Object.entries(manifest.batteries).map(([bid,r])=>[bid,r.summary.length,r.initial_capacity,r.eol_cycle,manifest.source]));
  refreshSweep();
}
function download(name,data,type){const url=URL.createObjectURL(new Blob([data],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('download-result').addEventListener('click',()=>{if(!current)return;const c=current;const row={battery:c.battery,cycle:c.cycle,model:c.modelKey,training:c.training,predicted_soh:c.result.soh,predicted_rul:c.result.rul,observed_soh:c.truth.soh,observed_rul:c.truth.rul,...c.cfg};download('battery_prediction.csv',Object.keys(row).join(',')+'\n'+Object.values(row).join(',')+'\n','text/csv');});
$('download-experiments').addEventListener('click',()=>{if(bench)download('battery_experiments.json',JSON.stringify(bench,null,2),'application/json');});
$('signal').addEventListener('change',redrawLive);
for(const id of ['factor','sweep-model','metric','tolerance'])$(id).addEventListener('change',()=>{if(id==='metric')$('tolerance').value=$('metric').value.startsWith('soh')?3:15;refreshSweep();});
for(const button of document.querySelectorAll('[role="tab"]'))button.addEventListener('click',()=>{
  for(const tab of document.querySelectorAll('[role="tab"]')){const selected=tab===button;tab.setAttribute('aria-selected',String(selected));$(tab.getAttribute('aria-controls')).hidden=!selected;}
  redrawLive();refreshSweep();
});
$('preset').addEventListener('change',()=>{
  const p=$('preset').value;if(p==='custom')return;
  for(const [id,value] of Object.entries({step:1,noise:0,missing:0,window:100,rounding:0,segment:'early'}))$(id).value=value;$('burst').checked=false;
  if(p==='noise')$('noise').value=50;
  if(p==='partial')$('window').value=50;
  if(p==='stress'){for(const [id,value] of Object.entries({step:10,noise:50,missing:50,window:50}))$(id).value=value;}
});
$('controls').addEventListener('input',e=>{if(!['battery','model','training','cycle','preset'].includes(e.target.id))$('preset').value='custom';labels();clearTimeout(timer);timer=setTimeout(update,120);});
$('controls').addEventListener('change',()=>{clearTimeout(timer);update();});
let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{redrawLive();refreshSweep();},100);});
try{[manifest,bench]=await Promise.all([get('./data/manifest.json'),get('./data/benchmarks.json')]);fillExperiments();await update();}
catch(e){status('Could not load the dashboard data. '+e.message+' Refresh to retry.',true);}
