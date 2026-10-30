"""Chapter 14: Mill Lane or Eastgate? A Huff gravity model of a fictional town.

Part of the companion code for The Retail Data Science Playbook (Shaon Biswas, 2026).
All data are synthetic and generated below with a fixed seed. Run from the repository root:
    python case_studies/ch14_site_choice_huff.py
"""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from common import *
import pandas as pd, statsmodels.formula.api as smf
rng=np.random.default_rng(14)
# ---- Ch14 Huff: town of Ashby Vale (fictional), grid of 400 neighbourhoods
g=np.linspace(0,10,20);XX,YY=np.meshgrid(g,g);pts=np.c_[XX.ravel(),YY.ravel()]
pop=np.exp(-((pts[:,0]-4.5)**2+(pts[:,1]-5)**2)/12)*1800+rng.uniform(100,400,len(pts));spend=pop*4.4   # £ grocery spend per week
stores={'Brightwell\'s High St':((3,6),1200,'B'),'Brightwell\'s Retail Park':((7.5,3),2600,'B'),'Discounter':((5.5,5.5),1800,'C'),'Big Four superstore':((2,2),4500,'C')}
cands={'Site A (Mill Lane)':((5,8),2000),'Site B (Eastgate)':((8.5,8),2000)}
alpha,lam=1.0,2.0
def shares(st):
    att=np.array([[size**alpha/ (np.hypot(*(pts-np.array(xy)).T)+.3)**lam for xy,size,_ in [st[k][:3] for k in st]]]).squeeze()
    return att/att.sum(0)
def sales(st):
    P=shares(st);return {k:float((P[i]*spend).sum()) for i,k in enumerate(st)}
now=sales(stores);out14=dict(now=now)
for c,(xy,size) in cands.items():
    st=dict(stores);st[c]=(xy,size,'B');s=sales(st)
    own_now=sum(v for k,v in now.items() if stores[k][2]=='B');own_new=sum(v for k,v in s.items() if k in stores and stores[k][2]=='B')+s[c]
    out14[c]=dict(new_store=s[c],cannibal=own_now-(own_new-s[c]),net_gain=own_new-own_now,after=s)
dump('cs14',out14);print(json.dumps({k:(v if k=='now' else {kk:vv for kk,vv in v.items() if kk!='after'}) for k,v in out14.items()},indent=1,default=float))
f,ax=plt.subplots(1,2,figsize=(5.6,2.6))
for a,(c,(xy,size)) in zip(ax,cands.items()):
    st=dict(stores);st[c]=(xy,size,'B');P=shares(st)
    a.imshow(P[-1].reshape(20,20),origin='lower',extent=[0,10,0,10],cmap='Greens',vmin=0,vmax=.6)
    for k,(p,_,t) in st.items():a.scatter(*p,s=30 if k!=c else 60,marker='*' if k==c else 'o',color=A if k==c else (GD if t=='B' else T),zorder=3)
    a.set_title(f"{c}: £{out14[c]['new_store']/1000:.0f}k/wk, net +£{out14[c]['net_gain']/1000:.0f}k",fontsize=7);a.set_xticks([]);a.set_yticks([])
save(f,'cs14_huff')
