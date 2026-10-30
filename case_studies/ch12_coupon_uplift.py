"""Chapter 12: churn risk vs uplift targeting for a randomised loyalty coupon.

Part of the companion code for The Retail Data Science Playbook (Shaon Biswas, 2026).
All data are synthetic and generated below with a fixed seed. Run from the repository root:
    python case_studies/ch12_coupon_uplift.py
"""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from common import *
import pandas as pd, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
rng=np.random.default_rng(12); N=120000
rec=rng.gamma(2,9,N); freq=rng.gamma(2.5,1.6,N); spend=rng.lognormal(3.6,.5,N); online=rng.random(N)<.3
fuel=rng.random(N)<.25; tenure=rng.gamma(3,1.2,N); comp=rng.random(N)<.35   # competitor discounter opened nearby
z=-1.5+0.05*rec-0.35*freq+0.9*comp-0.4*online-0.3*fuel-0.15*tenure+0.03*rec*comp+rng.normal(0,.5,N)
p0=1/(1+np.exp(-z))
# coupon effect: reduces lapse mainly for mid-frequency, price-sensitive members near a competitor; does little for loyal or gone
tau=-(0.12*comp+0.03)*((p0>0.12)&(p0<0.45))  # persuadables: mid risk; lost causes (p0>0.45) ignore the coupon
treat=rng.random(N)<.5
lapse=rng.random(N)<np.clip(p0+tau*treat,0,1)
X=pd.DataFrame(dict(recency=rec,frequency=freq,spend=spend,online=online.astype(int),fuel=fuel.astype(int),tenure=tenure,competitor=comp.astype(int)))
idx=rng.permutation(N);tr,te=idx[:90000],idx[90000:]
ctrl_tr=tr[~treat[tr]]
lr=LogisticRegression(max_iter=1000).fit(X.iloc[ctrl_tr],lapse[ctrl_tr]);gb=lgb.LGBMClassifier(n_estimators=400,learning_rate=.03,num_leaves=31,verbose=-1).fit(X.iloc[ctrl_tr],lapse[ctrl_tr])
ctrl_te=te[~treat[te]]
auc_lr=roc_auc_score(lapse[ctrl_te],lr.predict_proba(X.iloc[ctrl_te])[:,1]);auc_gb=roc_auc_score(lapse[ctrl_te],gb.predict_proba(X.iloc[ctrl_te])[:,1])
# T-learner uplift
m1=lgb.LGBMClassifier(n_estimators=300,learning_rate=.03,num_leaves=15,min_child_samples=100,verbose=-1).fit(X.iloc[tr[treat[tr]]],lapse[tr[treat[tr]]])
m0=lgb.LGBMClassifier(n_estimators=300,learning_rate=.03,num_leaves=15,min_child_samples=100,verbose=-1).fit(X.iloc[tr[~treat[tr]]],lapse[tr[~treat[tr]]])
up=m0.predict_proba(X.iloc[te])[:,1]-m1.predict_proba(X.iloc[te])[:,1]   # reduction in lapse probability
risk=gb.predict_proba(X.iloc[te])[:,1]
# evaluate targeting policies on the randomised test half: incremental retained per 1000 targeted
T=treat[te];Y=lapse[te]
def incr(order,frac):
    k=int(len(order)*frac);sel=order[:k];t=T[sel];y=Y[sel]
    return (y[~t].mean()-y[t].mean())*k  # retained customers if all k get coupon
fracs=np.linspace(.05,1,20);o_up=np.argsort(-up);o_risk=np.argsort(-risk);o_rand=rng.permutation(len(te))
cur={n:[incr(o,f) for f in fracs] for n,o in [('uplift',o_up),('risk',o_risk),('random',o_rand)]}
val=480;cpn=5*0.6   # value of a retained member over a year; coupon cost (redemption 60%)
def profit(o,frac): k=int(len(o)*frac);return incr(o,frac)*val-k*cpn
best={n:max(((f,profit(o,f)) for f in fracs),key=lambda x:x[1]) for n,o in [('uplift',o_up),('risk',o_risk)]}
out=dict(N=N,lapse_rate=lapse[~treat].mean(),ate=lapse[~treat].mean()-lapse[treat].mean(),auc_lr=auc_lr,auc_gb=auc_gb,
 top30_uplift=incr(o_up,.3),top30_risk=incr(o_risk,.3),all=incr(o_up,1.0),best_uplift=best['uplift'],best_risk=best['risk'],profit_all=profit(o_up,1.0),test_n=len(te),val=val,cpn=cpn)
dump('cs12',out);print(json.dumps(out,indent=1,default=float))
f,a=fig(5.6,2.6)
for n,c,lab in [('uplift',G,'rank by predicted uplift'),('risk',A,'rank by churn risk'),('random',GR,'random')]:a.plot(fracs*100,cur[n],color=c,lw=1.6 if n=='uplift' else 1.2,label=lab)
a.set_xlabel('% of 30,000 test members given the coupon');a.set_ylabel('Extra members retained');a.set_title('Targeting the persuadable, not the most at risk');a.legend();save(f,'cs12_qini')
