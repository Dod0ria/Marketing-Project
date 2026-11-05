# Scope justification and model sequence for the LG TV conversion project.
# Run from Marketing-Project: python scope_justification_and_models.py
# Requires pandas, numpy, statsmodels. Reads ../data/01_LG_Electronics_Appliances/electronics_market_behavior.csv
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.discrete.conditional_models import ConditionalLogit

pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 30)
pd.set_option('display.float_format', lambda v: f'{v:,.3f}')

DATA_DIR = Path('..') / 'data' / '01_LG_Electronics_Appliances'
ev = pd.read_csv(DATA_DIR / 'electronics_market_behavior.csv')
ev['event_time'] = pd.to_datetime(ev['event_time'], utc=True)
KEY = ['user_id', 'session_id', 'product_id']

# ---------------------------------------------------------------------------
# 1. Consideration table for ALL categories and brands (same rules as the TV cohort)
# ---------------------------------------------------------------------------
views = ev[ev['event_type'] == 'view']
first_views = (views.sort_values('event_time', kind='stable').drop_duplicates(KEY)
               [KEY + ['brand', 'product_category', 'event_time', 'price']]
               .rename(columns={'event_time': 'first_view', 'price': 'initial_price'}))
last_purchase = ev[ev['event_type'] == 'purchase'].groupby(KEY)['event_time'].max().rename('last_purchase')
carts = ev[ev['event_type'] == 'cart'].merge(first_views[KEY + ['first_view']], on=KEY)
carts = carts[carts['event_time'] > carts['first_view']]
first_cart = carts.groupby(KEY)['event_time'].min().rename('first_cart')
cohort = first_views.join(last_purchase, on=KEY).join(first_cart, on=KEY)
cohort['Y'] = (cohort['last_purchase'] > cohort['first_view']).astype(int)
cohort['has_cart'] = cohort['first_cart'].notna().astype(int)
cohort['purchase_after_cart'] = (cohort['last_purchase'] > cohort['first_cart']).astype(int)
cohort['month'] = cohort['first_view'].dt.strftime('%Y-%m')


def kpis(df, by):
    g = df.groupby(by).agg(rows=('Y', 'size'), users=('user_id', 'nunique'), products=('product_id', 'nunique'),
                           carts=('has_cart', 'sum'), purchases_after_cart=('purchase_after_cart', 'sum'),
                           purchases=('Y', 'sum'), median_price=('initial_price', 'median'))
    g['view_to_cart_pct'] = 100 * g['carts'] / g['rows']
    g['cart_to_purchase_pct'] = 100 * g['purchases_after_cart'] / g['carts'].replace(0, np.nan)
    g['view_to_purchase_pct'] = 100 * g['purchases'] / g['rows']
    return g


# ---------------------------------------------------------------------------
# 2. Why TVs: LG against Samsung and against all other brands, in every category
# ---------------------------------------------------------------------------
rows = []
for cat, sub in cohort.groupby('product_category'):
    lg, sam = sub[sub['brand'] == 'lg'], sub[sub['brand'] == 'samsung']
    rest = sub[sub['brand'] != 'lg']
    se = lambda x: np.sqrt(x['Y'].mean() * (1 - x['Y'].mean()) / len(x))
    rows.append({'category': cat, 'category_rows': len(sub), 'category_median_price': sub['initial_price'].median(),
                 'lg_rows': len(lg), 'lg_purchases': lg['Y'].sum(),
                 'lg_share_of_rows_pct': 100 * len(lg) / len(sub), 'lg_share_of_purchases_pct': 100 * lg['Y'].sum() / sub['Y'].sum(),
                 'lg_v2p_pct': 100 * lg['Y'].mean(), 'samsung_v2p_pct': 100 * sam['Y'].mean(), 'others_v2p_pct': 100 * rest['Y'].mean(),
                 'gap_vs_samsung_pp': 100 * (lg['Y'].mean() - sam['Y'].mean()),
                 'z_vs_samsung': (lg['Y'].mean() - sam['Y'].mean()) / np.sqrt(se(lg) ** 2 + se(sam) ** 2),
                 'gap_vs_others_pp': 100 * (lg['Y'].mean() - rest['Y'].mean())})
category_table = pd.DataFrame(rows).set_index('category')
print('\n=== LG position by category ===')
print(category_table)

print('\n=== Stage gaps LG minus Samsung by category (pp) ===')
stage = []
for cat, sub in cohort.groupby('product_category'):
    k = kpis(sub, ['brand'])
    stage.append({'category': cat, 'view_to_cart': k.loc['lg', 'view_to_cart_pct'] - k.loc['samsung', 'view_to_cart_pct'],
                  'cart_to_purchase': k.loc['lg', 'cart_to_purchase_pct'] - k.loc['samsung', 'cart_to_purchase_pct'],
                  'view_to_purchase': k.loc['lg', 'view_to_purchase_pct'] - k.loc['samsung', 'view_to_purchase_pct'],
                  'lg_carts': k.loc['lg', 'carts'], 'samsung_carts': k.loc['samsung', 'carts']})
print(pd.DataFrame(stage).set_index('category'))

# ---------------------------------------------------------------------------
# 3. Why Samsung: price positioning of every TV brand relative to LG
# ---------------------------------------------------------------------------
tv_all = cohort[cohort['product_category'] == 'tv']
print('\n=== TV initial-price quantiles by brand ===')
print(tv_all.groupby('brand')['initial_price'].quantile([.05, .25, .5, .75, .95]).unstack().round(0))
lo, hi = tv_all.loc[tv_all['brand'] == 'lg', 'initial_price'].quantile([.05, .95])
print(f'\nShare of each brand\'s TV rows inside LG\'s 5th-95th price range ({lo:.0f}-{hi:.0f}):')
print(tv_all.groupby('brand')['initial_price'].apply(lambda s: 100 * s.between(lo, hi).mean()).round(1))
print('\nBrands present per category:')
print(cohort.groupby('product_category')['brand'].unique())

# ---------------------------------------------------------------------------
# 4. Within-shopper comparison in every category (users who viewed both brands)
# ---------------------------------------------------------------------------
rows = []
for cat, sub in cohort.groupby('product_category'):
    s2 = sub[sub['brand'].isin(['lg', 'samsung'])]
    nb = s2.groupby('user_id')['brand'].nunique()
    dual = s2[s2['user_id'].isin(nb[nb == 2].index)]
    k = kpis(dual, ['brand'])
    rows.append({'category': cat, 'dual_brand_users': dual['user_id'].nunique(),
                 'lg_v2p_pct': k.loc['lg', 'view_to_purchase_pct'], 'samsung_v2p_pct': k.loc['samsung', 'view_to_purchase_pct'],
                 'gap_pp': k.loc['lg', 'view_to_purchase_pct'] - k.loc['samsung', 'view_to_purchase_pct']})
print('\n=== Same shoppers, both brands: view-to-purchase by category ===')
print(pd.DataFrame(rows).set_index('category'))

# ---------------------------------------------------------------------------
# 5. TV sample: shopper composition and decomposition of the raw gap
# ---------------------------------------------------------------------------
tv = tv_all[tv_all['brand'].isin(['lg', 'samsung'])].copy()
shopper_type = tv.groupby('user_id')['brand'].agg(lambda x: 'comparison' if x.nunique() == 2 else 'single_brand')
tv['shopper_type'] = tv['user_id'].map(shopper_type)
comp = tv.groupby(['brand', 'shopper_type']).agg(rows=('Y', 'size'), users=('user_id', 'nunique'), v2p_pct=('Y', 'mean')).reset_index()
comp['v2p_pct'] *= 100
comp['share_of_brand_rows_pct'] = 100 * comp['rows'] / comp.groupby('brand')['rows'].transform('sum')
print('\n=== TV: shopper composition by brand ===')
print(comp)
lg_rate = tv.loc[tv['brand'] == 'lg', 'Y'].mean()
sam_rate = tv.loc[tv['brand'] == 'samsung', 'Y'].mean()
w_sam = comp[comp['brand'] == 'samsung'].set_index('shopper_type')['share_of_brand_rows_pct'] / 100
r_lg = comp[comp['brand'] == 'lg'].set_index('shopper_type')['v2p_pct'] / 100
counterfactual = (w_sam * r_lg).sum()
print(f'Raw gap {100 * (lg_rate - sam_rate):+.3f} pp = composition {100 * (lg_rate - counterfactual):+.3f} pp '
      f'+ within-type {100 * (counterfactual - sam_rate):+.3f} pp '
      f'(composition share {100 * (lg_rate - counterfactual) / (lg_rate - sam_rate):.0f}%)')

# ---------------------------------------------------------------------------
# 6. Pre-exposure shopper covariates (measured strictly before the first view)
# ---------------------------------------------------------------------------
tv['LG'] = (tv['brand'] == 'lg').astype(int)
tv['log_price'] = np.log(tv['initial_price'])
tv['november'] = (tv['month'] == '2019-11').astype(int)
hist = views.merge(tv[['user_id', 'session_id', 'product_id', 'brand', 'first_view']].rename(columns={'product_id': 'key_product', 'brand': 'key_brand'}),
                   on=['user_id', 'session_id'])
hist = hist[hist['event_time'] < hist['first_view']]
hist['other_brand_tv'] = (hist['product_category'] == 'tv') & hist['brand'].isin(['lg', 'samsung']) & (hist['brand'] != hist['key_brand'])
pre = hist.groupby(['user_id', 'session_id', 'key_product']).agg(
    pre_products=('product_id', 'nunique'), pre_brands=('brand', 'nunique'), other_brand_tv=('other_brand_tv', 'max')
).reset_index().rename(columns={'key_product': 'product_id'})
tv = tv.merge(pre, on=KEY, how='left')
tv[['pre_products', 'pre_brands']] = tv[['pre_products', 'pre_brands']].fillna(0)
tv['other_brand_tv'] = tv['other_brand_tv'].fillna(False).astype(int)
session_start = ev.groupby(['user_id', 'session_id'])['event_time'].min().reset_index(name='session_start').sort_values('session_start')
session_start['prior_sessions'] = session_start.groupby('user_id').cumcount()
tv = tv.merge(session_start[['user_id', 'session_id', 'prior_sessions']], on=['user_id', 'session_id'])
print('\n=== Pre-exposure covariates, mean by brand ===')
print(tv.groupby('brand')[['pre_products', 'pre_brands', 'other_brand_tv', 'prior_sessions']].mean())

# ---------------------------------------------------------------------------
# 7. Model sequence, outcome Y = purchase strictly after first view, SE clustered by user
# ---------------------------------------------------------------------------
def fit_logit(formula, data, label):
    m = smf.logit(formula, data=data).fit(disp=False, cov_type='cluster', cov_kwds={'groups': data['user_id']})
    lg = 100 * m.predict(data.assign(LG=1)).mean()
    sam = 100 * m.predict(data.assign(LG=0)).mean()
    print(f'{label:48s} LG coef {m.params["LG"]:+.3f} (SE {m.bse["LG"]:.3f})  adjusted gap {lg - sam:+.3f} pp  N={len(data):,}')
    return m

print('\n=== Model sequence ===')
m1 = fit_logit('Y ~ LG', tv, 'M1 brand only')
m2 = fit_logit('Y ~ LG + log_price + november', tv, 'M2 + log initial price + month')
m3 = fit_logit('Y ~ LG + log_price + november + pre_products + pre_brands + other_brand_tv + prior_sessions', tv,
               'M3 + pre-exposure shopper covariates')
print(m3.summary2().tables[1].round(4))

dual = tv[tv['shopper_type'] == 'comparison']
fe = dual[dual.groupby('user_id')['Y'].transform('nunique') == 2]
m4 = ConditionalLogit(fe['Y'], fe[['LG', 'log_price', 'november']], groups=fe['user_id']).fit()
print(f'\nM4 user fixed effects (conditional logit), comparison shoppers with varying Y: {fe["user_id"].nunique()} users, {len(fe)} rows')
print(pd.DataFrame({'coef': m4.params, 'SE': m4.bse, 'p': m4.pvalues, 'odds_ratio': np.exp(m4.params)}).round(4))

tv['price_quintile'] = pd.qcut(tv['initial_price'], 5, labels=False)
m5 = smf.logit('Y ~ LG * C(price_quintile) + november', data=tv).fit(disp=False, cov_type='cluster', cov_kwds={'groups': tv['user_id']})
print('\nM5 LG x price quintile (quintile 0 = cheapest):')
print(m5.summary2().tables[1].loc[[i for i in m5.params.index if 'LG' in i]].round(4))
print('\nRaw view-to-purchase by price quintile and brand (%):')
print((100 * tv.groupby(['price_quintile', 'brand'])['Y'].mean()).unstack().round(2))