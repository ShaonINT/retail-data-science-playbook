# The Retail Data Science Playbook: companion code

Code and data generators for **The Retail Data Science Playbook** by Shaon Biswas: how to build, evaluate and choose models for pricing, forecasting, promotions, inventory and customers, with 41 algorithms and nine case studies from a fictional UK grocer, Brightwell's.

The book is free to download at **[imshaon.com](https://imshaon.com)**.

Every number in the book's case studies and in the Part I model contest is produced by this repository. The data are synthetic, generated with fixed seeds, so running the code reproduces the printed tables and charts exactly with the pinned library versions.

## Quick start

```bash
git clone https://github.com/ShaonINT/retail-data-science-playbook.git
cd retail-data-science-playbook
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py            # about 4 minutes on a laptop
```

Run one chapter with `python run_all.py ch09`, or open the matching notebook in `notebooks/`.

## Case studies

| Ch. | Question | Code | Result in the book |
|---|---|---|---|
| 6 | Comparing models and choosing the winner | [`ch06_model_contest.py`](case_studies/ch06_model_contest.py) · [notebook](notebooks/ch06_model_contest.ipynb) | Ridge 32.7% vs LightGBM 32.8% WAPE; no difference detected |
| 7 | Which shelves will be empty next week? | [`ch07_out_of_stock_ranking.py`](case_studies/ch07_out_of_stock_ranking.py) · [notebook](notebooks/ch07_out_of_stock_ranking.ipynb) | XGBoost finds an OOS in 42% of checks vs 23% for the current rule |
| 8 | One forecast does not fit every line | [`ch08_forecast_routing.py`](case_studies/ch08_forecast_routing.py) · [notebook](notebooks/ch08_forecast_routing.ipynb) | LightGBM for fast movers, Croston for slow movers, analogue curve for launches |
| 9 | The price of a bag of coffee | [`ch09_coffee_pricing.py`](case_studies/ch09_coffee_pricing.py) · [notebook](notebooks/ch09_coffee_pricing.ipynb) | Elasticity -2.65 (naive) vs -1.79 (double ML); true −1.8 |
| 10 | The chocolate promotion that lost money | [`ch10_promotion_profit.py`](case_studies/ch10_promotion_profit.py) · [notebook](notebooks/ch10_promotion_profit.ipynb) | +182% units, −£2,665 profit; display DiD +6.2% |
| 11 | How many loaves to bake at 6 a.m. | [`ch11_bakery_newsvendor.py`](case_studies/ch11_bakery_newsvendor.py) · [notebook](notebooks/ch11_bakery_newsvendor.ipynb) | +£11,096 gross profit over 8 weeks |
| 12 | Who should get the £5 coupon? | [`ch12_coupon_uplift.py`](case_studies/ch12_coupon_uplift.py) · [notebook](notebooks/ch12_coupon_uplift.ipynb) | Top 30% by uplift keeps 94% of the full-campaign effect |
| 13 | The 'you might also need' row | [`ch13_basket_rules.py`](case_studies/ch13_basket_rules.py) · [notebook](notebooks/ch13_basket_rules.ipynb) | Recall@5 81% vs 44% for best-sellers |
| 14 | Mill Lane or Eastgate? | [`ch14_site_choice_huff.py`](case_studies/ch14_site_choice_huff.py) · [notebook](notebooks/ch14_site_choice_huff.ipynb) | Mill Lane net +£152k/wk vs Eastgate +£105k/wk |
| 15 | Did the new shelf-edge labels work? | [`ch15_shelf_label_trial_cuped.py`](case_studies/ch15_shelf_label_trial_cuped.py) · [notebook](notebooks/ch15_shelf_label_trial_cuped.ipynb) | CI narrows from [-7.4%, 22.8%] to [-0.4%, 4.2%] |
| 16 | The soft-drinks forecast drifts | [`ch16_drift_monitoring.py`](case_studies/ch16_drift_monitoring.py) · [notebook](notebooks/ch16_drift_monitoring.ipynb) | Alert in week 107; challenger WAPE 23.8% vs 26.2% |

Outputs go to `results/` (JSON with every number quoted in the book) and `figures/` (SVG charts).

## What is reproducible, and what is illustrative

- **Executed results:** all case-study tables and charts, and the Part I model contest. They come from the scripts above.
- **Illustrative:** the short code snippets and the charts on the two-page algorithm reference pages. They show how a method works, not measured results.

## Repository layout

```
case_studies/   one script per case study (data generator + models + charts)
notebooks/      the same code as notebooks, with results displayed
results/        JSON outputs quoted in the book
figures/        SVG charts used in the book
run_all.py      regenerates everything; used by the GitHub Actions workflow
```

## Licence and citation

The code is released under the MIT licence. The book text is © 2026 Shaon Biswas and is not part of this repository.

If you use this work, please cite it using `CITATION.cff` (GitHub shows a "Cite this repository" button).

## Author

Shaon Biswas · [imshaon.com](https://imshaon.com) · [ORCID 0000-0002-3085-2517](https://orcid.org/0000-0002-3085-2517)

Brightwell's, its people and its data are fictional. Any resemblance to a real company or person is coincidental.
