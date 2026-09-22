const M = window.__MODEL__;
const $ = id => document.getElementById(id);
const state = { age:58, sex:1, cp:4, trestbps:158, chol:285, fbs:0, restecg:2,
  thalach:118, exang:1, oldpeak:2.6, slope:2, ca:2, thal:7 };
const PRETTY = { age:'Age', trestbps:'Resting blood pressure', chol:'Cholesterol',
  thalach:'Max heart rate', oldpeak:'ST depression', ca:'Blocked vessels (fluoroscopy)',
  rpp:'Rate-pressure product', hr_reserve:'Heart-rate reserve', sex:'Male sex',
  fbs:'High fasting blood sugar', exang:'Exercise-induced angina' };
const CP = { 'cp_1.0':'Chest pain: typical angina','cp_2.0':'Chest pain: atypical angina',
  'cp_3.0':'Chest pain: non-anginal','cp_4.0':'Chest pain: asymptomatic',
  'restecg_0.0':'ECG: normal','restecg_1.0':'ECG: ST-T abnormality','restecg_2.0':'ECG: LV hypertrophy',
  'slope_1.0':'ST slope: upsloping','slope_2.0':'ST slope: flat','slope_3.0':'ST slope: downsloping',
  'thal_3.0':'Thallium: normal','thal_6.0':'Thallium: fixed defect','thal_7.0':'Thallium: reversible defect' };
Object.assign(PRETTY, CP);
const pretty = n => PRETTY[n] || n;

function read() {
  state.age=+$('age').value; state.trestbps=+$('trestbps').value;
  state.chol=+$('chol').value; state.thalach=+$('thalach').value;
  state.oldpeak=+$('oldpeak').value; state.ca=+$('ca').value;
  state.cp=+$('cp').value; state.restecg=+$('restecg').value;
  state.slope=+$('slope').value; state.thal=+$('thal').value;
  state.sex=parseInt($('sex').dataset.value,10);
  state.fbs=parseInt($('fbs').dataset.value,10);
  state.exang=parseInt($('exang').dataset.value,10);
}
function vector(s) {
  const rpp=(s.trestbps*s.thalach)/100, hr=s.thalach/(220-s.age);
  return { age:s.age,trestbps:s.trestbps,chol:s.chol,thalach:s.thalach,oldpeak:s.oldpeak,ca:s.ca,rpp:rpp,hr_reserve:hr,
    'cp_1.0':s.cp===1?1:0,'cp_2.0':s.cp===2?1:0,'cp_3.0':s.cp===3?1:0,'cp_4.0':s.cp===4?1:0,
    'restecg_0.0':s.restecg===0?1:0,'restecg_1.0':s.restecg===1?1:0,'restecg_2.0':s.restecg===2?1:0,
    'slope_1.0':s.slope===1?1:0,'slope_2.0':s.slope===2?1:0,'slope_3.0':s.slope===3?1:0,
    'thal_3.0':s.thal===3?1:0,'thal_6.0':s.thal===6?1:0,'thal_7.0':s.thal===7?1:0, sex:s.sex,fbs:s.fbs,exang:s.exang };
}

function predict(s) {
  const raw=vector(s), names=M.feature_names;
  let z=M.intercept; const contrib=[];
  for (let i=0;i<names.length;i++){
    const n=names[i]; let x=raw[n]||0;
    const ni=M.num_cols.indexOf(n), bi=M.bin_cols.indexOf(n);
    if (ni>=0) x=(x-M.num_mean[ni])/M.num_scale[ni];
    else if (bi>=0) x=(x-M.bin_mean[bi])/M.bin_scale[bi];
    const c=M.coef[i]*x; z+=c; contrib.push({name:n,value:c});
  }
  const cz=M.calibrator_coef[0]*z+M.calibrator_intercept;
  return {p:1/(1+Math.exp(-cz)),contrib:contrib};
}

function render() {
  read();
  const r=predict(state), p=r.p, contrib=r.contrib, thr=M.threshold;
  const high=p>=thr, mid=!high&&p>=0.25;
  $('verdict').className='verdict '+(high?'high':mid?'mid':'low');
  $('vtitle').textContent=high?'HIGH RISK':mid?'MODERATE RISK':'LOW RISK';
  $('vdesc').textContent=high?'This profile meets the referral threshold. Invasive coronary angiography should be considered.'
    :mid?'Below the referral threshold but worth monitoring. Continue risk-factor control and follow-up.'
    :'This profile is well below the referral threshold. Routine non-invasive monitoring is appropriate.';
  $('vprob').textContent=(p*100).toFixed(1)+'%'; $('sprob').textContent=(p*100).toFixed(1)+'%';
  $('sthr').textContent=(thr*100).toFixed(1)+'%';
  $('marker').style.left='calc('+(p*100).toFixed(1)+'% - 2px)';
  $('thr').style.left=(thr*100).toFixed(1)+'%';
  $('thrlabel').textContent='threshold '+(thr*100).toFixed(1)+'%';
  const top=contrib.slice().sort((a,b)=>Math.abs(b.value)-Math.abs(a.value)).slice(0,10);
  const scale=Math.max(...top.map(c=>Math.abs(c.value)),1e-6);
  $('contrib').innerHTML=top.map(c=>{
    const w=Math.abs(c.value)/scale*50, up=c.value>0;
    return '<div class="row"><div class="name">'+pretty(c.name)+'</div>'
      +'<div class="bwrap"><div class="mid"></div>'
      +'<div class="bar '+(up?'up':'dn')+'" style="width:'+w.toFixed(1)+'%"></div></div>'
      +'<div class="amt">'+(c.value>=0?'+':'')+c.value.toFixed(3)+'</div></div>';
  }).join('');
  const ups=top.filter(c=>c.value>0).slice(0,3).map(c=>pretty(c.name));
  const dns=top.filter(c=>c.value<0).slice(0,3).map(c=>pretty(c.name));
  $('summary').innerHTML=(ups.length?'<b>Increasing risk:</b> '+ups.join(', '):'')
    +(ups.length&&dns.length?'<br>':'')+(dns.length?'<b>Decreasing risk:</b> '+dns.join(', '):'');
}

function simulate() {
  read();
  const bp=+$('cbp').value, ch=+$('cchol').value;
  $('cbpv').textContent=bp+' mmHg'; $('ccholv').textContent=ch+' mg/dl';
  const before=predict(state).p, after=predict(Object.assign({},state,{trestbps:bp,chol:ch})).p;
  const col=p=>p>=0.5?'#ef4444':p>=0.25?'#f59e0b':'#22c55e';
  $('b1').style.height=Math.max(before*220,4)+'px'; $('b1').style.background=col(before);
  $('b2').style.height=Math.max(after*220,4)+'px'; $('b2').style.background=col(after);
  $('n1').textContent=(before*100).toFixed(1)+'%'; $('n2').textContent=(after*100).toFixed(1)+'%';
  const d=before-after, note=$('cfnote');
  if (d>0.001){ note.className='note ok'; note.textContent='Modelled risk falls '+(d*100).toFixed(1)+' percentage points ('+(d/before*100).toFixed(1)+'% relative reduction).'; }
  else if (d<-0.001){ note.className='note bad'; note.textContent='These targets would raise modelled risk by '+(Math.abs(d)*100).toFixed(1)+' points.'; }
  else { note.className='note info'; note.textContent='These targets leave modelled risk essentially unchanged.'; }
}

function sample(which) {
  const s=which==='high'
    ?{age:58,sex:1,cp:4,trestbps:158,chol:285,fbs:0,restecg:2,thalach:118,exang:1,oldpeak:2.6,slope:2,ca:2,thal:7}
    :{age:41,sex:0,cp:2,trestbps:118,chol:168,fbs:0,restecg:0,thalach:176,exang:0,oldpeak:0,slope:1,ca:0,thal:3};
  ['age','trestbps','chol','thalach','oldpeak','ca','cp','restecg','slope','thal'].forEach(k=>{
    $(k).value=s[k]; const lbl=$(k+'v'); if (lbl) lbl.textContent=s[k]; });
  setToggle('sex',s.sex); setToggle('fbs',s.fbs); setToggle('exang',s.exang);
  $('cbp').value=Math.min(s.trestbps,140); $('cchol').value=Math.min(s.chol,200);
  render(); simulate();
}
function setToggle(id,v){
  const b=$(id); b.dataset.value=v;
  b.querySelectorAll('button').forEach(x=>x.classList.toggle('on',+x.dataset.v===v));
}
function tab(id){
  document.querySelectorAll('.tabs button').forEach(b=>b.classList.toggle('on',b.dataset.t===id));
  document.querySelectorAll('.tabpane').forEach(p=>p.classList.toggle('on',p.id==='pane-'+id));
}

function init(){
  ['age','trestbps','chol','thalach','oldpeak','ca'].forEach(k=>{
    const i=$(k);
    i.addEventListener('input',()=>{ $(k+'v').textContent=i.value; render(); });
  });
  ['cp','restecg','slope','thal'].forEach(k=>$(k).addEventListener('change',render));
  ['sex','fbs','exang'].forEach(k=>{
    $(k).querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{ setToggle(k,+b.dataset.v); render(); }));
  });
  ['cbp','cchol'].forEach(k=>$(k).addEventListener('input',simulate));
  document.querySelectorAll('.tabs button').forEach(b=>b.addEventListener('click',()=>tab(b.dataset.t)));
  render(); simulate();
}
document.addEventListener('DOMContentLoaded', init);
