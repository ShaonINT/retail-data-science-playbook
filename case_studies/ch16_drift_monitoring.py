"""Chapter 16: forecast drift after a competitor opens - alerts, PSI and a retrained challenger.

Part of the companion code for The Retail Data Science Playbook (Shaon Biswas, 2026).
All data are synthetic and generated below with a fixed seed. Run from the repository root:
    python case_studies/ch16_drift_monitoring.py
"""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from common import *
import pandas as pd, lightgbm as lgb
rng=np.random.default_rng(16); nI=60; W=140; tr_end=92
rows=[]
for i in range(nI):
    lvl=rng.uniform(80,400)
    for t in range(W):
        temp=14+8*np.sin(2*np.pi*(t-10)/52)+rng.normal(0,2)
        comp=1 if t>=106 else 0                                                         # discounter opens week 106
        promo=int(rng.random()<.1)
        mu=lvl*(1+0.04*(temp-14))*(1+0.8*promo)*(1-0.25*comp)
        rows.append(dict(item=i,t=t,temp=temp,promo=promo,comp=comp,y=rng.poisson(max(mu,1)*rng.gamma(60,1/60))))
D=pd.DataFrame(rows).sort_values(['item','t']);g=D.groupby('item').y
D['lag4']=g.shift(8);D['lag52']=g.shift(52);D['rm']=g.transform(lambda s:s.shift(8).rolling(8).mean());D['yoy']=D.rm/D.groupby('item').rm.shift(52)
feats=['temp','promo','lag4','lag52','rm']
trn=D[(D.t<tr_end)&D.lag52.notna()]
m=lgb.LGBMRegressor(n_estimators=400,learning_rate=.03,num_leaves=31,verbose=-1).fit(trn[feats],trn.y)   # competitor not a feature: it did not exist
D['f']=m.predict(D[feats]);D['naive']=D.lag52
wk=D[D.t>=tr_end].groupby('t').apply(lambda x:pd.Series(dict(m=np.abs(x.y-x.f).sum()/x.y.sum(),n=np.abs(x.y-x.naive).sum()/x.y.sum(),bias=(x.f-x.y).sum()/x.y.sum())))
roll=wk.rolling(4).mean()
# PSI on the forecast inputs: distribution of recent sales level (rm) vs training reference
ref=trn.yoy.dropna();edges=np.quantile(ref,np.linspace(0,1,11));edges[0]=-np.inf;edges[-1]=np.inf
def psi(x):
    a=np.histogram(ref,edges)[0]/len(ref);b=np.histogram(x,edges)[0]/len(x);a=np.clip(a,1e-4,None);b=np.clip(b,1e-4,None);return float(np.sum((b-a)*np.log(b/a)))
ps=D[D.t>=tr_end].groupby('t').yoy.apply(lambda x:psi(x.dropna()))
val_w=wk.loc[:105,'m'].mean()
alert=[t for t in roll.index if t>=tr_end+3 and (roll.loc[t,'m']>val_w+0.05 or abs(roll.loc[t,'bias'])>0.10)]
first_alert=int(alert[0]) if alert else None
# retrain with competitor flag and recent data at week 112 (challenger), evaluate weeks 113-129
trn2=D[(D.t<122)&D.lag52.notna()];m2=lgb.LGBMRegressor(n_estimators=400,learning_rate=.03,num_leaves=31,verbose=-1).fit(trn2[feats+['comp']],trn2.y)
post=D[D.t>=123];w_old=np.abs(post.y-post.f).sum()/post.y.sum();w_new=np.abs(post.y-m2.predict(post[feats+['comp']])).sum()/post.y.sum();w_nv=np.abs(post.y-post.naive).sum()/post.y.sum()
pre=D[(D.t>=tr_end)&(D.t<106)]
out=dict(val_wape=float(val_w),train_end=tr_end,comp_week=106,heat=[114,120],first_alert=first_alert,
 wape_pre=float(np.abs(pre.y-pre.f).sum()/pre.y.sum()),naive_pre=float(np.abs(pre.y-pre.naive).sum()/pre.y.sum()),
 bias_after=float(wk.loc[106:121,'bias'].mean()),psi_before=float(ps.loc[:105].mean()),psi_after=float(ps.loc[110:121].mean()),psi_max=float(ps.max()),
 post_old=float(w_old),post_new=float(w_new),post_naive=float(w_nv))
dump('cs16',out);print(json.dumps(out,indent=1));print(roll.loc[100:139:3].round(3).to_string())
f,(a1,a2)=plt.subplots(2,1,figsize=(5.6,3.3),sharex=True)
a1.plot(roll.index,roll.m*100,color=G,lw=1.6,label='model, rolling 4-week WAPE');a1.plot(roll.index,roll.n*100,color=GR,lw=1.2,ls='--',label='seasonal naive');a1.axhline((val_w+0.05)*100,color=T,lw=.6,ls=':');a1.text(93,(val_w+0.055)*100,'alert line: launch WAPE + 5 pts',fontsize=6,color=M)
for a in (a1,a2):
    a.axvline(106,color=A,ls=':',lw=1)
a1.text(106.5,a1.get_ylim()[1]*0.95 if False else roll.m.max()*100,'discounter\nopens',fontsize=6,color=A,va='top')
if first_alert: a1.axvline(first_alert,color=T,lw=.8);a1.text(first_alert+.4,roll.n.max()*100,'alert',fontsize=6)
a1.set_ylabel('WAPE (%)');a1.legend(fontsize=6,loc='upper left');a1.set_title('Soft drinks: forecast error against the baseline')
a2.plot(ps.index,ps.values,color=B,lw=1.4);a2.axhline(1.0,color=A,ls='--',lw=.8);a2.text(93,1.05,'action level (set from normal weeks)',fontsize=6,color=A)
a2.set_ylabel('PSI');a2.set_xlabel('Week');a2.set_title('Population stability of the sales-versus-last-year input')
save(f,'cs16_drift')
