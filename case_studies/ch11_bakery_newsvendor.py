"""Chapter 11: how many loaves to bake - newsvendor replay against two ordering rules.

Part of the companion code for The Retail Data Science Playbook (Shaon Biswas, 2026).
All data are synthetic and generated below with a fixed seed. Run from the repository root:
    python case_studies/ch11_bakery_newsvendor.py
"""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from common import *
import pandas as pd
rng=np.random.default_rng(11); S=30; D=7*20  # 12 weeks history + 8 weeks evaluation
price,cost=2.20,0.70; under=price-cost; over=cost  # no salvage (unsold loaves donated)
cr=under/(under+over)
dow=np.array([.85,.8,.85,.95,1.25,1.45,.85])
lvl=rng.uniform(25,70,S); rows=[]
for s in range(S):
    for d in range(D):
        mu=lvl[s]*dow[d%7]*(1+0.002*d); dem=rng.negative_binomial(8,8/(8+mu))
        rows.append(dict(store=s,day=d,dow=d%7,demand=dem))
df=pd.DataFrame(rows); hist_end=7*12
def run(rule):
    sold=waste=lost=prof=0;avail=[]
    for s in range(S):
        g=df[df.store==s].set_index('day')
        for d in range(hist_end,D):
            q=rule(g,d); dem=g.demand[d]; sl=min(q,dem)
            sold+=sl;waste+=q-sl;lost+=max(dem-q,0);prof+=sl*price-q*cost;avail.append(dem<=q)
    return dict(waste_pct=waste/(sold+waste),avail=np.mean(avail),lost=lost,profit=prof,sold=sold)
# current rule: same day last week + 10%
cur=lambda g,d: int(np.ceil(g.demand[d-7]*1.10))
# average rule: mean of last 4 same weekdays (a 'good forecast' but ordering the mean)
mean4=lambda g,d: int(round(np.mean([g.demand[d-7*k] for k in range(1,5)])))
# newsvendor: critical-ratio quantile of last 8 same weekdays (scaled empirical) 
def nv(g,d):
    past=np.array([g.demand[d-7*k] for k in range(1,9)]); return int(np.ceil(np.quantile(past,cr)))
out=dict(price=price,cost=cost,cr=cr,stores=S,eval_days=D-hist_end)
for n,r in [('Last week + 10%',cur),('Average of last 4 weeks',mean4),('Newsvendor quantile',nv)]:
    out[n]=run(r)
dump('cs11',out);print(json.dumps(out,indent=1,default=float))
names=['Last week + 10%','Average of last 4 weeks','Newsvendor quantile']
f,(a1,a2)=plt.subplots(1,2,figsize=(5.6,2.5))
a1.bar(range(3),[out[n]['waste_pct']*100 for n in names],color=[GR,B,G],width=.6);a1.set_xticks(range(3));a1.set_xticklabels(['Last week\n+10%','Average of\nlast 4','Newsvendor\nquantile'],fontsize=6.5);a1.set_title('Waste (% of loaves baked)')
a2.bar(range(3),[out[n]['profit']/1000 for n in names],color=[GR,B,G],width=.6);a2.set_xticks(range(3));a2.set_xticklabels(['Last week\n+10%','Average of\nlast 4','Newsvendor\nquantile'],fontsize=6.5);a2.set_title('Gross profit, 8 weeks (£k)')
lo=min(out[n]['profit'] for n in names)/1000;a2.set_ylim(lo*0.95,None)
save(f,'cs11_bake')
