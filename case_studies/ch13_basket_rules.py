"""Chapter 13: association rules and an offline recall@5 test for basket suggestions.

Part of the companion code for The Retail Data Science Playbook (Shaon Biswas, 2026).
All data are synthetic and generated below with a fixed seed. Run from the repository root:
    python case_studies/ch13_basket_rules.py
"""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from common import *
import pandas as pd, itertools, collections
rng=np.random.default_rng(13)
items=['milk','bread','eggs','butter','bananas','pasta','pasta sauce','parmesan','garlic','onions','minced beef','tortilla wraps','salsa','cheddar','avocados','limes',
 'tonic','gin','crisps','lager','nappies','baby wipes','coffee','oat milk','croissants','orange juice','chicken','rice','curry paste','coconut milk','naan','yoghurt','granola','berries']
I={n:i for i,n in enumerate(items)}
missions={'top-up':(['milk','bread','eggs','butter','bananas'],.32),'pasta night':(['pasta','pasta sauce','parmesan','garlic','onions','minced beef'],.12),
 'fajitas':(['tortilla wraps','salsa','cheddar','avocados','limes','chicken','onions'],.08),'party':(['tonic','gin','crisps','lager','limes'],.08),
 'baby':(['nappies','baby wipes','milk','bananas'],.07),'breakfast':(['coffee','oat milk','croissants','orange juice','yoghurt','granola','berries'],.15),
 'curry':(['chicken','rice','curry paste','coconut milk','naan','onions'],.08),'mixed':(items,.10)}
mn=list(missions);mp=np.array([missions[m][1] for m in mn]);mp/=mp.sum()
pop=np.array([5,5,3,3,4,2,2,1,2,3,2,1,1,2,1,1,1,1,3,2,1,1,2,1,1,2,2,2,1,1,1,2,1,1],float)
def basket():
    m=mn[rng.choice(len(mn),p=mp)];pool=missions[m][0];k=min(len(pool),rng.integers(2,6))
    w=np.array([pop[I[x]] for x in pool]);b=set(rng.choice(pool,k,replace=False,p=w/w.sum()))
    if rng.random()<.5: b.add(items[rng.choice(len(items),p=pop/pop.sum())])
    return sorted(b)
B=[basket() for _ in range(60000)];n=len(B)
cnt=collections.Counter(x for b in B for x in b);pair=collections.Counter(p for b in B for p in itertools.combinations(b,2))
rules=[]
for (a,b),c in pair.items():
    sup=c/n
    if c<300: continue
    for x,y in [(a,b),(b,a)]:
        conf=c/cnt[x];lift=conf/(cnt[y]/n);rules.append(dict(antecedent=x,consequent=y,support=sup,confidence=conf,lift=lift))
R=pd.DataFrame(rules).sort_values('lift',ascending=False)
top=R.drop_duplicates(subset=['support']).head(8)
# recommendation test: hide one item from each test basket, recommend top-5
train,test=B[:50000],B[50000:]
cnt2=collections.Counter(x for b in train for x in b);pair2=collections.Counter(p for b in train for p in itertools.combinations(b,2))
popular=[x for x,_ in cnt2.most_common()]
def co(b,k=5):
    sc=collections.Counter()
    for x in b:
        for (a,c),v in pair2.items():
            if a==x and c not in b: sc[c]+=v/cnt2[x]/(cnt2[c]/50000)**.5
            elif c==x and a not in b: sc[a]+=v/cnt2[x]/(cnt2[a]/50000)**.5
    return [x for x,_ in sc.most_common(k)]
hp=hc=tot=0
for b in test:
    if len(b)<2: continue
    h=b[rng.integers(len(b))];rest=[x for x in b if x!=h];tot+=1
    hp+=h in [x for x in popular if x not in rest][:5];hc+=h in co(rest)
out=dict(baskets=n,rules=top.round(3).to_dict('records'),recall5_pop=hp/tot,recall5_co=hc/tot,test=tot)
dump('cs13',out);print(json.dumps(out,indent=1,default=float)[:2500])
