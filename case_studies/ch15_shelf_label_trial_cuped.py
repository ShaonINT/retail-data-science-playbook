"""Chapter 15: a randomised store trial analysed with and without CUPED.

Part of the companion code for The Retail Data Science Playbook (Shaon Biswas, 2026).
All data are synthetic and generated below with a fixed seed. Run from the repository root:
    python case_studies/ch15_shelf_label_trial_cuped.py
"""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from common import *
import pandas as pd, statsmodels.formula.api as smf
# ---- Ch15: shelf-edge price label trial, 80 stores, CUPED
rng=np.random.default_rng(15);S=80;pre=rng.lognormal(10.6,.35,S);treat=rng.permutation(np.r_[np.ones(40),np.zeros(40)]).astype(bool)
post=pre*rng.normal(1.0,.05,S)*(1+0.025*treat)
d=pd.DataFrame(dict(pre=pre,post=post,t=treat.astype(int)))
naive=smf.ols('post ~ t',d).fit();theta=np.cov(d.post,d.pre)[0,1]/d.pre.var();d['cuped']=d.post-theta*(d.pre-d.pre.mean());cup=smf.ols('cuped ~ t',d).fit()
m=d[d.t==0].post.mean()
out15=dict(stores=S,naive=naive.params['t']/m,naive_ci=[x/m for x in naive.conf_int().loc['t']],cuped=cup.params['t']/m,cuped_ci=[x/m for x in cup.conf_int().loc['t']],
 var_red=1-d.cuped.var()/d.post.var(),naive_p=naive.pvalues['t'],cuped_p=cup.pvalues['t'],weekly_sales=float(m))
# conformal coverage check on forecast residuals (from a simple model) 
res=rng.standard_t(4,3000)*30;cal,test=res[:1500],res[1500:]
q=np.quantile(np.abs(cal),np.ceil(1501*.9)/1500);gauss=1.645*cal.std()
out15.update(conf_cov=float(np.mean(np.abs(test)<=q)),gauss_cov=float(np.mean(np.abs(test)<=1.645*np.std(cal))),conf_q=float(q),gauss_q=float(gauss))
dump('cs15',out15);print(json.dumps(out15,indent=1,default=float))
f,a=fig(5.6,2.3)
for i,(lab,e,ci,c) in enumerate([('Simple difference',out15['naive'],out15['naive_ci'],A),('CUPED-adjusted',out15['cuped'],out15['cuped_ci'],G)]):
    a.errorbar(e*100,i,xerr=[[(e-ci[0])*100],[(ci[1]-e)*100]],fmt='o',color=c,capsize=4,lw=1.5)
a.axvline(0,color=M,lw=.6);a.set_yticks([0,1]);a.set_yticklabels(['Simple difference','CUPED-adjusted']);a.set_ylim(-.6,1.6);a.set_xlabel('Effect on weekly store sales (%), 95% CI');a.set_title('Same trial, same stores: CUPED narrows the interval');save(f,'cs15_cuped')
