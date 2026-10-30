"""Chapter 8: one forecast does not fit every line (fast movers, slow movers, new launches).

Part of the companion code for The Retail Data Science Playbook (Shaon Biswas, 2026).
All data are synthetic and generated below with a fixed seed. Run from the repository root:
    python case_studies/ch08_forecast_routing.py
"""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from common import *
import pandas as pd, warnings, lightgbm as lgb
from statsmodels.tsa.holtwinters import ExponentialSmoothing
warnings.filterwarnings('ignore')
rng=np.random.default_rng(8); W=130; H=13   # 117 weeks history, test last 13 weeks (4 origins? use single holdout per origin rolling)
def season(t,ph): return np.sin(2*np.pi*(t+ph)/52)
series=[]
for i in range(40):   # fast movers, promo-driven
    lvl=rng.uniform(150,600);ph=rng.uniform(0,52);t=np.arange(W)
    promo=(rng.random(W)<0.12).astype(int); price=np.where(promo,0.75,1.0)
    mu=lvl*(1+0.18*season(t,ph))*(1+0.002*t)*np.exp(-1.6*np.log(price))
    y=rng.poisson(mu*rng.gamma(30,1/30,W)); series.append(dict(id=f'F{i}',seg='fast',y=y,promo=promo,start=0))
for i in range(40):   # slow, intermittent
    p=rng.uniform(.15,.4);size=rng.uniform(1,3);t=np.arange(W)
    y=np.where(rng.random(W)<p,rng.poisson(size,W)+1,0); series.append(dict(id=f'S{i}',seg='slow',y=y,promo=np.zeros(W,int),start=0))
for i in range(20):   # new launches: only last 21 weeks exist (8 history + 13 test)
    lvl=rng.uniform(80,300);ph=rng.uniform(0,52);t=np.arange(W)
    ramp=1-np.exp(-(t-(W-21)+1)/9); promo=(rng.random(W)<0.1).astype(int)
    mu=lvl*np.clip(ramp,0,None)*(1+0.18*season(t,ph))*np.exp(-1.6*np.log(np.where(promo,0.75,1)))
    y=rng.poisson(np.clip(mu,0,None)); y[:W-21]=0; series.append(dict(id=f'N{i}',seg='new',y=y,promo=promo,start=W-21))
tr_end=W-H
def mase(y,f,ytr,m=52):
    sc=np.mean(np.abs(ytr[m:]-ytr[:-m])) if len(ytr)>m+5 else np.mean(np.abs(np.diff(ytr)))
    return np.mean(np.abs(y-f))/max(sc,1e-6)
def wape(y,f): return np.sum(np.abs(y-f))/max(np.sum(y),1e-6)
# --- methods
def snaive(s):
    y=s['y'];
    if s['start']>0: return np.repeat(y[tr_end-1],H)            # naive last value for new
    return y[tr_end-52:tr_end-52+H].astype(float)
def ets(s):
    y=s['y'][s['start']:tr_end].astype(float)
    try:
        if len(y)>=104: m=ExponentialSmoothing(y+1,trend='add',damped_trend=True,seasonal='add',seasonal_periods=52).fit()
        else: m=ExponentialSmoothing(y+1,trend='add',damped_trend=True).fit()
        return np.clip(m.forecast(H)-1,0,None)
    except Exception: return np.repeat(y[-4:].mean(),H)
def croston(s,a=.1):
    y=s['y'][s['start']:tr_end].astype(float); z=None;p=None;q=1
    for v in y:
        if v>0:
            z=v if z is None else z+a*(v-z); p=q if p is None else p+a*(q-p); q=1
        else: q+=1
    return np.repeat(0 if z is None else (1-a/2)*z/p,H)
# global LightGBM (direct horizon via lag>=H features)
rows=[]
for s in series:
    y=s['y'].astype(float);
    for t in range(max(s['start'],0),W):
        rows.append(dict(id=s['id'],seg=s['seg'],t=t,y=y[t],promo=s['promo'][t],woy=t%52,age=t-s['start'],
            lag13=y[t-13] if t-13>=s['start'] else np.nan,lag26=y[t-26] if t-26>=s['start'] else np.nan,lag52=y[t-52] if t-52>=s['start'] else np.nan,
            rm13=y[max(s['start'],t-26):t-13].mean() if t-13>s['start'] else np.nan))
D=pd.DataFrame(rows);D['segc']=D.seg.astype('category');feats=['promo','woy','age','lag13','lag26','lag52','rm13','segc']
trn=D[D.t<tr_end];m=lgb.LGBMRegressor(objective='tweedie',tweedie_variance_power=1.3,n_estimators=600,learning_rate=.03,num_leaves=31,min_child_samples=20,verbose=-1).fit(trn[feats],trn.y)
D['lgb']=m.predict(D[feats])
def gbm(s): return D[(D.id==s['id'])&(D.t>=tr_end)].lgb.values
# analogue for new launches: average launch curve of the other new items, scaled to week 1-8 level
def analog(s):
    others=[o for o in series if o['seg']=='new' and o['id']!=s['id']]
    curves=np.array([o['y'][o['start']:o['start']+21]/max(o["y"][o["start"]:o["start"]+8].mean(),1) for o in others]).mean(0)
    return curves[8:21]*s['y'][s['start']:tr_end].mean()
M={'Seasonal naive':snaive,'ETS':ets,'Croston (SBA)':croston,'LightGBM (global)':gbm,'Analogue launch curve':analog}
res={}
for seg in ['fast','slow','new']:
    res[seg]={}
    for name,fn in M.items():
        if name=='Analogue launch curve' and seg!='new': continue
        if name=='Croston (SBA)' and seg!='slow': continue
        sc=[];wa_n=0;wa_d=0
        for s in series:
            if s['seg']!=seg: continue
            f=fn(s);y=s['y'][tr_end:];ytr=s['y'][s['start']:tr_end].astype(float)
            sc.append(mase(y,f,ytr));wa_n+=np.abs(y-f).sum();wa_d+=y.sum()
        res[seg][name]=dict(mase=float(np.mean(sc)),wape=float(wa_n/wa_d))
dump('cs8',res)
for k,v in res.items(): print(k,{n:(round(x['wape']*100,1),round(x['mase'],2)) for n,x in v.items()})
# figure: WAPE by segment
f,a=fig(5.6,2.6);segs=['fast','slow','new'];lab={'fast':'Fast movers\n(promo-driven)','slow':'Slow movers\n(intermittent)','new':'New launches\n(8 weeks history)'}
names=list(M);cols=[GR,B,A,G,GD];w=.16
for j,n in enumerate(names):
    xs=[i+(j-2)*w for i,s in enumerate(segs) if n in res[s]];vs=[res[s][n]['wape']*100 for s in segs if n in res[s]]
    a.bar(xs,vs,w,color=cols[j],label=n)
a.set_xticks(range(3));a.set_xticklabels([lab[s] for s in segs]);a.set_ylabel('WAPE (%), lower is better');a.set_title('No single model wins every segment');a.legend(ncol=3,fontsize=6,loc='upper left');a.set_ylim(0,max(res['slow'][n]['wape'] for n in res['slow'])*100*1.35)
save(f,'cs8_seg')
