"""Chapter 6: five forecasting models compared with walk-forward validation, paired tests and a decision matrix.

Part of the companion code for The Retail Data Science Playbook (Shaon Biswas, 2026).
All data are synthetic and generated below with a fixed seed. Run from the repository root:
    python case_studies/ch06_model_contest.py
"""
# Worked case study for Part I: compare forecasting models with walk-forward validation on synthetic retail data.
import numpy as np, pandas as pd, warnings, json, time, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; (ROOT / 'results').mkdir(exist_ok=True)
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import Ridge
import lightgbm as lgb
warnings.filterwarnings('ignore')
rng = np.random.default_rng(2026)
N, W = 30, 156
rows = []
for i in range(N):
    base = rng.uniform(40, 300); e = rng.uniform(1.2, 2.6); amp = rng.uniform(.05, .35); tr = rng.uniform(-.001, .003)
    p_reg = rng.uniform(1.0, 5.0)
    for w in range(W):
        promo = int(rng.random() < .1); disc = rng.uniform(.15, .35) if promo else 0.0
        price = p_reg * (1 - disc) * (1 + rng.normal(0, .01))
        season = 1 + amp * np.sin(2 * np.pi * (w % 52) / 52 - 1.0) + (0.25 if w % 52 in (49, 50) else 0)
        mu = base * season * (1 + tr) ** w * (price / p_reg) ** (-e)
        y = rng.negative_binomial(8, 8 / (8 + mu))
        rows.append((i, w, y, price, p_reg, promo))
df = pd.DataFrame(rows, columns=['sku', 'week', 'y', 'price', 'p_reg', 'promo'])
df['price_ratio'] = df.price / df.p_reg
df['woy'] = df.week % 52
H = 4
g = df.groupby('sku')['y']
for L in (4, 5, 6, 8, 12, 52):
    df[f'lag{L}'] = g.shift(L)
df['roll4_12'] = g.transform(lambda s: s.shift(4).rolling(9).mean())
df['xmas'] = df.woy.isin([49, 50]).astype(int)
feats = ['lag4', 'lag5', 'lag6', 'lag8', 'lag12', 'lag52', 'roll4_12', 'price_ratio', 'promo', 'woy', 'xmas']

origins = list(range(104, W - H + 1, 4))  # 13 folds
res = []
t0 = time.time()
for o in origins:
    train = df[df.week < o]; test = df[(df.week >= o) & (df.week < o + H)]
    tr = train.dropna(subset=feats)
    gbm = lgb.LGBMRegressor(objective='tweedie', n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=20, verbose=-1)
    gbm.fit(tr[feats], tr.y)
    rid = Ridge(alpha=1.0).fit(tr[feats], tr.y)
    p_gbm = gbm.predict(test[feats]); p_rid = rid.predict(test[feats])
    for k, (idx, row) in enumerate(test.iterrows()):
        pass
    test = test.assign(gbm=p_gbm, ridge=p_rid)
    for s in range(N):
        hist = train[train.sku == s].y.values
        tt = test[test.sku == s]
        snaive = hist[-52:][:H] if True else None
        snaive = np.array([hist[len(hist) - 52 + h] for h in range(H)])
        try:
            ets = ExponentialSmoothing(hist, trend='add', damped_trend=True, seasonal='add', seasonal_periods=52).fit(optimized=True)
            p_ets = ets.forecast(H)
        except Exception:
            p_ets = snaive
        scale = np.mean(np.abs(hist[52:] - hist[:-52]))
        for h in range(H):
            r = tt.iloc[h]
            res.append(dict(origin=o, sku=s, h=h + 1, y=r.y, promo=r.promo, scale=scale, snaive=snaive[h], ets=p_ets[h], ridge=r.ridge, gbm=r.gbm))
R = pd.DataFrame(res)
R['ensemble'] = (R.ets + R.gbm) / 2
models = ['snaive', 'ets', 'ridge', 'gbm', 'ensemble']
names = {'snaive': 'Seasonal naive', 'ets': 'ETS (Holt-Winters)', 'ridge': 'Ridge regression', 'gbm': 'LightGBM', 'ensemble': 'ETS + LightGBM average'}
summary = []
for m in models:
    e = R[m] - R.y
    summary.append(dict(model=names[m], key=m, WAPE=np.abs(e).sum() / R.y.sum() * 100, Bias=e.sum() / R.y.sum() * 100,
                        MASE=np.mean(np.abs(e) / R.scale), RMSE=np.sqrt(np.mean(e ** 2)),
                        WAPE_promo=np.abs(e[R.promo == 1]).sum() / R.y[R.promo == 1].sum() * 100,
                        fold_sd=R.assign(ae=np.abs(e)).groupby('origin').apply(lambda d: d.ae.sum() / d.y.sum() * 100).std()))
S = pd.DataFrame(summary)
# fold-level WAPE for boxplot
fold = {m: R.assign(ae=np.abs(R[m] - R.y)).groupby('origin').apply(lambda d: d.ae.sum() / d.y.sum() * 100).tolist() for m in models}
# Diebold-Mariano on per-(origin) mean absolute loss differences: gbm vs ets, ensemble vs gbm
from scipy.stats import t as tdist
def dm(a, b):
    d = (np.abs(R[a] - R.y) - np.abs(R[b] - R.y)).groupby(R.origin).mean().values
    n = len(d); stat = d.mean() / (d.std(ddof=1) / np.sqrt(n)); p = 2 * (1 - tdist.cdf(abs(stat), n - 1))
    return dict(stat=float(stat), p=float(p), n=n)
DM = {'gbm_vs_ets': dm('gbm', 'ets'), 'ens_vs_gbm': dm('ensemble', 'gbm'), 'gbm_vs_snaive': dm('gbm', 'snaive')}
# example sku plot data
ex = 7
hist = df[(df.sku == ex)][['week', 'y', 'promo']]
exR = R[R.sku == ex].sort_values(['origin', 'h'])
out = dict(summary=S.round(3).to_dict('records'), fold=fold, dm=DM, n_skus=N, weeks=W, folds=len(origins), H=H, rows=len(R),
           ex_hist=hist.values.tolist(), ex_pred=exR[['origin', 'h', 'y', 'snaive', 'ets', 'gbm']].values.tolist())
json.dump(out, open(ROOT / 'results' / 'ch06_model_contest.json', 'w'))
print(S.round(2).to_string()); print(DM); print('secs', round(time.time() - t0))
DM['gbm_vs_ridge']=dm('gbm','ridge'); DM['ridge_vs_ets']=dm('ridge','ets')
out['dm']=DM; json.dump(out, open(ROOT / 'results' / 'ch06_model_contest.json', 'w')); print(DM['gbm_vs_ridge'],DM['ridge_vs_ets'])
