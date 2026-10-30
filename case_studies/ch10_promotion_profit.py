"""Chapter 10: the chocolate promotion profit bridge and the display-trial difference-in-differences.

Part of the companion code for The Retail Data Science Playbook (Shaon Biswas, 2026).
All data are synthetic and generated below with a fixed seed. Run from the repository root:
    python case_studies/ch10_promotion_profit.py
"""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from common import *
import pandas as pd, statsmodels.formula.api as smf
rng=np.random.default_rng(10)
W=60; t=np.arange(W); promo_wk=[40,41]
base=1800*(1+0.1*np.sin(2*np.pi*t/52))
y=base*rng.gamma(80,1/80,W); sib=1500*(1+0.1*np.sin(2*np.pi*t/52))*rng.gamma(80,1/80,W)
lift=np.zeros(W);lift[40]=1.75;lift[41]=1.55
y=y*(1+lift); pf=np.zeros(W);pf[42]=-0.22;pf[43]=-0.10; y=y*(1+pf)
sib[40]*=0.80;sib[41]*=0.82
price=2.50;cost=1.30;promo_price=1.25;funding=0.30   # supplier funds 30p per unit on promo
# baseline: regression on non-promo weeks with seasonality (weeks 0-39)
d=pd.DataFrame(dict(t=t,y=y,s=np.sin(2*np.pi*t/52),c=np.cos(2*np.pi*t/52)))
m=smf.ols('y ~ s + c',d[t<40]).fit();bl=m.predict(d)
d_sib=pd.DataFrame(dict(t=t,y=sib,s=d.s,c=d.c));ms=smf.ols('y ~ s + c',d_sib[t<40]).fit();bls=ms.predict(d_sib)
promo_units=y[40:42].sum();base_units=bl[40:42].sum();incr=promo_units-base_units
pull=(y[42:44]-bl[42:44]).sum();cann=(sib[40:42]-bls[40:42]).sum()
supplier_uplift=(y[40:42].mean()/y[36:40].mean()-1)
margin_promo=promo_price-cost+funding; margin_reg=price-cost
profit_promo=promo_units*margin_promo; profit_base=base_units*margin_reg
net=profit_promo-profit_base+pull*margin_reg+cann*(2.20-1.10)
out=dict(promo_units=promo_units,base_units=base_units,incr=incr,incr_pct=incr/base_units,supplier_uplift=supplier_uplift,pull=pull,cann=cann,
 margin_promo=margin_promo,margin_reg=margin_reg,profit_promo=profit_promo,profit_base=profit_base,pull_cost=pull*margin_reg,cann_cost=cann*1.10,net=net)
# DiD: gondola-end display in 40 stores (from week 20), 160 control stores, 32 weeks
S=200;Wd=32;rows=[]
lvl=rng.lognormal(4,.4,S);treated=np.zeros(S,int);treated[:40]=1;lvl[:40]*=1.15  # trial stores are larger
for s in range(S):
    for w in range(Wd):
        post=int(w>=20);mu=lvl[s]*(1+0.004*w)*(1+0.08*np.sin(2*np.pi*w/52))*(1+0.062*treated[s]*post)
        rows.append(dict(store=s,week=w,treated=treated[s],post=post,units=rng.poisson(mu*rng.gamma(25,1/25))))
D=pd.DataFrame(rows);D['ly']=np.log(D.units+1)
did=smf.ols('ly ~ treated:post + C(store) + C(week)',D).fit(cov_type='cluster',cov_kwds={'groups':D.store})
naive_cmp=D[(D.post==1)&(D.treated==1)].units.mean()/D[(D.post==1)&(D.treated==0)].units.mean()-1
ev=[]
for w in range(Wd):
    g=D[D.week==w];ev.append(g[g.treated==1].ly.mean()-g[g.treated==0].ly.mean())
ev=np.array(ev)-np.mean(ev[:20])
out.update(did=did.params['treated:post'],did_ci=list(did.conf_int().loc['treated:post']),naive_cmp=naive_cmp)
dump('cs10',out);print(json.dumps(out,indent=1,default=float))
f,(a1,a2)=plt.subplots(1,2,figsize=(5.6,2.6))
a1.plot(t[30:50],y[30:50],color=T,lw=1.2,label='actual');a1.plot(t[30:50],bl[30:50],color=G,ls='--',lw=1.2,label='baseline')
a1.fill_between(t[40:42],bl[40:42],y[40:42],color=G,alpha=.2,label='incremental');a1.fill_between(t[42:44],y[42:44],bl[42:44],color=A,alpha=.3,label='pull-forward dip')
a1.set_title('Sharing bags: promo weeks 40–41');a1.set_xlabel('Week');a1.set_ylabel('Units');a1.legend(fontsize=5.8)
a2.axhline(0,color=M,lw=.5);a2.plot(range(Wd),ev*100,'o-',color=G,ms=2.5,lw=1);a2.axvline(19.5,color=GR,ls=':',lw=1);a2.text(20.3,ev.min()*100,'displays\ninstalled',fontsize=6,color=M)
a2.set_title('Display trial: treated minus control');a2.set_xlabel('Week');a2.set_ylabel('Difference (%)');save(f,'cs10_promo')
