// Numerical inference over the original Python-trained model parameters.
export const median = a => {
  if (!a.length) return NaN;
  const v=[...a].sort((a,b)=>a-b),n=v.length;
  return n%2?v[(n-1)/2]:(v[n/2-1]+v[n/2])/2;
};
const mean = a => a.reduce((s,v)=>s+v,0)/a.length;
export function decodeTrace(buffer) {
  if(buffer.byteLength%32) throw Error('Invalid sensor data file.');
  const view=new DataView(buffer),rows=[];
  for(let i=0;i<buffer.byteLength;i+=32) rows.push([0,8,16,24].map(j=>view.getFloat64(i+j,true)));
  return rows;
}
function rng(seed) {
  let x=seed>>>0;
  return ()=>{x+=0x6D2B79F5;let t=x;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;};
}
export function degrade(raw,c) {
  const random=rng(c.seed??42);
  const normal=()=>Math.sqrt(-2*Math.log(Math.max(random(),1e-12)))*Math.cos(2*Math.PI*random());
  const n=Math.max(1,Math.floor(raw.length*c.window)),start=c.upper?Math.floor(raw.length*(1-c.window)):0;
  let rows=raw.slice(start,start+n).filter((_,i)=>i%c.step===0).map(r=>r.slice());
  if(c.missing>0){let keep=rows.map(()=>random()>=c.missing);if(keep.filter(Boolean).length<5)keep=keep.map((v,i)=>i<5||v);rows=rows.filter((_,i)=>keep[i]);}
  if(c.burst&&rows.length>30){const gap=Math.floor(rows.length*.15),begin=Math.floor(rows.length*.2+random()*rows.length*.5);rows.splice(begin,gap);}
  for(const [col,std,precision] of [[1,c.noise,c.rounding||0],[2,c.noise*2,0],[3,c.noise*25,0]]) {
    for(const row of rows){if(std>0)row[col]+=normal()*std;if(precision>0)row[col]=Math.round(row[col]/precision)*precision;}
  }
  return rows;
}
export function features(trace,cycle) {
  const f={cycle,charging_time_sec:NaN,v_mean:NaN,v_std:NaN,v_min:NaN,v_max:NaN,v_skew:NaN,t_max:NaN,t_rise:NaN,coulomb_capacity_ah:NaN,v_trans_time_sec:NaN};
  const rows=trace.filter(r=>Number.isFinite(r[0])).toSorted((a,b)=>a[0]-b[0]).filter((r,i,a)=>!i||r[0]!==a[i-1][0]);
  if(rows.length<3)return f;
  f.charging_time_sec=rows.at(-1)[0]-rows[0][0];
  const volts=rows.map(r=>r[1]).filter(Number.isFinite),temps=rows.map(r=>r[3]).filter(Number.isFinite);
  if(volts.length>=3){
    const avg=mean(volts),m2=mean(volts.map(v=>(v-avg)**2)),m3=mean(volts.map(v=>(v-avg)**3)),n=volts.length;
    Object.assign(f,{v_mean:avg,v_std:Math.sqrt(m2),v_min:Math.min(...volts),v_max:Math.max(...volts),v_skew:m2>1e-24?Math.sqrt(n*(n-1))/(n-2)*m3/(m2**1.5):0});
  }
  if(temps.length>=2){f.t_max=Math.max(...temps);f.t_rise=Math.max(0,f.t_max-temps[0]);}
  const pairs=rows.filter(r=>Number.isFinite(r[2]));
  if(pairs.length>=2){
    const diffs=pairs.slice(1).map((r,i)=>r[0]-pairs[i][0]),limit=5*median(diffs);
    f.coulomb_capacity_ah=diffs.reduce((sum,dt,i)=>sum+(dt<=limit?.5*(Math.max(0,pairs[i][2])+Math.max(0,pairs[i+1][2]))*dt:0),0)/3600;
  }
  const valid=rows.filter(r=>Number.isFinite(r[1]));
  if(valid.length>1&&valid[0][1]<=3.8){const low=valid.find(r=>r[1]>=3.8),high=valid.find(r=>r[1]>=4.1);if(low&&high)f.v_trans_time_sec=Math.max(0,high[0]-low[0]);}
  return f;
}
function treePredict(tree,x) {
  let node=0;
  while(tree.left[node]!==-1)node=x[tree.feature[node]]<=tree.threshold[node]?tree.left[node]:tree.right[node];
  return tree.value[node];
}
function predictEstimator(m,x) {
  if(!m)return null;
  if(m.kind==='linear')return m.intercept+m.coef.reduce((sum,v,i)=>sum+v*x[i],0);
  // sklearn tree predictors evaluate float32 feature values after preprocessing.
  const treeX=x.map(Math.fround),sum=m.trees.reduce((total,t)=>total+treePredict(t,treeX),0);
  return m.kind==='rf'?sum/m.trees.length:m.base+m.rate*sum;
}
export function predict(model,f) {
  const x=model.features.map((name,i)=>((Number.isFinite(f[name])?f[name]:model.impute[i])-model.mean[i])/model.scale[i]);
  const rul=predictEstimator(model.rul,x);
  return {soh:predictEstimator(model.soh,x),rul:rul===null?null:Math.max(0,rul)};
}
