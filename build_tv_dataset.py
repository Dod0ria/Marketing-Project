# Builds the two working datasets for the LG TV project from the original behaviour file.
# Run from Marketing-Project: python 01_build_tv_dataset.py
# Reads  ../data/01_LG_Electronics_Appliances/electronics_market_behavior.csv
# Writes ../data/01_LG_Electronics_Appliances/tv_events_clean.csv     (event level, TV only, six brands)
#        ../data/01_LG_Electronics_Appliances/tv_considerations.csv   (one row per user x session x TV product)
# Column definitions are in tv_considerations_dictionary.md. Requires pandas and numpy.
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path('..') / 'data' / '01_LG_Electronics_Appliances'
KEY = ['user_id', 'session_id', 'product_id']
SIMILAR = np.log(1.2)  # a rival model counts as similar when its price is within +/-20% of the row's price

# ---------------------------------------------------------------------------------------------
# 1. Clean events: drop exact duplicates, parse times, keep TVs
# ---------------------------------------------------------------------------------------------
raw = pd.read_csv(DATA_DIR / 'electronics_market_behavior.csv')
ev = raw.drop_duplicates().copy()
ev['event_time'] = pd.to_datetime(ev['event_time'], utc=True)
for col in ['event_type', 'brand', 'product_category']:
    ev[col] = ev[col].str.strip().str.lower()
assert ev.notna().all().all() and (ev['price'] > 0).all()
assert ev['event_type'].isin(['view', 'cart', 'purchase']).all()
assert (ev.groupby('product_id')[['brand', 'product_category']].nunique() == 1).all().all()

tv_ev = ev[ev['product_category'] == 'tv'].sort_values(['event_time', 'user_id', 'session_id'], kind='stable')
tv_ev.to_csv(DATA_DIR / 'tv_events_clean.csv', index=False)

V = tv_ev[tv_ev['event_type'] == 'view'][['user_id', 'session_id', 'product_id', 'brand', 'event_time', 'price']]
C = tv_ev[tv_ev['event_type'] == 'cart'][KEY + ['event_time']]
P = tv_ev[tv_ev['event_type'] == 'purchase'][['user_id', 'session_id', 'product_id', 'brand', 'event_time']]

# ---------------------------------------------------------------------------------------------
# 2. One row per user x session x product, timed at the first view
# ---------------------------------------------------------------------------------------------
rows = (V.sort_values('event_time', kind='stable').drop_duplicates(KEY)
        .rename(columns={'event_time': 'first_view', 'price': 'initial_price'}).reset_index(drop=True))
rows['rid'] = np.arange(len(rows))
rows['brand_group'] = np.where(rows['brand'].isin(['lg', 'samsung']), rows['brand'], 'budget')
rows['month'] = rows['first_view'].dt.strftime('%Y-%m')
rows['november'] = (rows['month'] == '2019-11').astype(int)
rows['log_price'] = np.log(rows['initial_price'])

# outcome and funnel stages (all strictly after the first view, same session)
last_buy = P.groupby(KEY)['event_time'].max().rename('last_purchase')
first_buy_after = (P.merge(rows[KEY + ['first_view']], on=KEY).query('event_time > first_view')
                   .groupby(KEY)['event_time'].min().rename('first_purchase_after_view'))
first_cart = (C.merge(rows[KEY + ['first_view']], on=KEY).query('event_time > first_view')
              .groupby(KEY)['event_time'].min().rename('first_cart_after_view'))
rows = rows.join(last_buy, on=KEY).join(first_buy_after, on=KEY).join(first_cart, on=KEY)
rows['purchased'] = (rows['last_purchase'] > rows['first_view']).astype(int)
rows['carted'] = rows['first_cart_after_view'].notna().astype(int)
rows['purchased_after_cart'] = (rows['last_purchase'] > rows['first_cart_after_view']).astype(int)

# ---------------------------------------------------------------------------------------------
# 3. Shopper's state on arrival (only events BEFORE the first view, after the user's last TV purchase)
# ---------------------------------------------------------------------------------------------
pr = rows[['rid', 'user_id', 'first_view']].merge(P[['user_id', 'event_time']].rename(columns={'event_time': 'pt'}), on='user_id')
rows = rows.join(pr[pr['pt'] < pr['first_view']].groupby('rid')['pt'].max().rename('last_tv_purchase_before'), on='rid')
rows['tv_purchase_before'] = rows['last_tv_purchase_before'].notna().astype(int)

hist = rows[['rid', 'user_id', 'session_id', 'brand', 'first_view', 'initial_price', 'last_tv_purchase_before']].merge(
    V.rename(columns={'session_id': 'v_session', 'product_id': 'v_product', 'brand': 'v_brand', 'event_time': 'v_time', 'price': 'v_price'}),
    on='user_id')
hist = hist[(hist['v_time'] < hist['first_view'])
            & (hist['last_tv_purchase_before'].isna() | (hist['v_time'] > hist['last_tv_purchase_before']))]
hist['same_session'] = hist['v_session'] == hist['session_id']
hist['other_brand'] = hist['v_brand'] != hist['brand']
hist['similar_price'] = np.log(hist['v_price'] / hist['initial_price']).abs() <= SIMILAR
age = hist['first_view'] - hist['v_time']


def flag(mask, name):
    return hist[mask].groupby('rid').size().gt(0).rename(name)


s = hist[hist['same_session']]
rows = rows.join(s.groupby('rid')['v_product'].nunique().rename('earlier_tvs_in_visit'), on='rid')
for f in [flag(hist['same_session'] & hist['other_brand'], 'other_brand_before_in_visit'),
          flag(hist['same_session'] & hist['other_brand'] & hist['similar_price'], 'similar_rival_before_in_visit'),
          flag(hist['other_brand'] & (age <= pd.Timedelta(hours=48)), 'other_brand_before_48h'),
          flag(hist['other_brand'] & (age <= pd.Timedelta(days=7)), 'other_brand_before_7d'),
          flag(hist['other_brand'], 'other_brand_before_any')]:
    rows = rows.join(f, on='rid')
flags = ['other_brand_before_in_visit', 'similar_rival_before_in_visit', 'other_brand_before_48h',
         'other_brand_before_7d', 'other_brand_before_any']
rows[flags] = rows[flags].fillna(False).astype(int)
rows['earlier_tvs_in_visit'] = rows['earlier_tvs_in_visit'].fillna(0).astype(int)
rows['first_tv_in_visit'] = (rows['earlier_tvs_in_visit'] == 0).astype(int)
rows['position_in_visit'] = np.select(
    [rows['first_tv_in_visit'] == 1, rows['other_brand_before_in_visit'] == 0],
    ['first_tv_in_visit', 'later_same_brand_only_before'], 'later_other_brand_before')
start = rows['first_view'].min().normalize()
rows['window_48h_complete'] = (rows['first_view'] >= start + pd.Timedelta(hours=48)).astype(int)
rows['window_7d_complete'] = (rows['first_view'] >= start + pd.Timedelta(days=7)).astype(int)

# earlier sessions on the whole site (any category), counted by session start
ss = ev.groupby(['user_id', 'session_id'])['event_time'].min().reset_index(name='session_start').sort_values('session_start', kind='stable')
ss['prior_sessions'] = ss.groupby('user_id').cumcount()
rows = rows.merge(ss[['user_id', 'session_id', 'prior_sessions']], on=['user_id', 'session_id'], how='left')

# ---------------------------------------------------------------------------------------------
# 4. What happened after the page was opened (same session, strictly after the first view)
# ---------------------------------------------------------------------------------------------
after_v = rows[['rid', 'user_id', 'session_id', 'brand', 'first_view']].merge(
    V[['user_id', 'session_id', 'brand', 'event_time']].rename(columns={'brand': 'v_brand', 'event_time': 'v_time'}),
    on=['user_id', 'session_id'])
after_v = after_v[(after_v['v_time'] > after_v['first_view']) & (after_v['v_brand'] != after_v['brand'])]
rows = rows.join(after_v.groupby('rid')['v_time'].min().rename('first_other_brand_view_after'), on='rid')

after_p = rows[['rid', 'user_id', 'session_id', 'product_id', 'brand', 'first_view']].merge(
    P.rename(columns={'product_id': 'p_product', 'brand': 'p_brand', 'event_time': 'p_time'}), on=['user_id', 'session_id'])
after_p = after_p[after_p['p_time'] > after_p['first_view']]
other = after_p[after_p['p_brand'] != after_p['brand']].copy()
other['p_group'] = np.where(other['p_brand'].isin(['lg', 'samsung']), other['p_brand'], 'budget')
rows = rows.join(other.groupby('rid').size().gt(0).rename('bought_other_brand_after'), on='rid')
rows = rows.join(other.sort_values('p_time').groupby('rid')['p_group'].first().rename('other_brand_bought_group'), on='rid')
rows = rows.join(after_p[(after_p['p_brand'] == after_p['brand']) & (after_p['p_product'] != after_p['product_id'])]
                 .groupby('rid').size().gt(0).rename('bought_other_model_same_brand_after'), on='rid')
for c in ['bought_other_brand_after', 'bought_other_model_same_brand_after']:
    rows[c] = rows[c].fillna(False).astype(int)
rows['viewed_other_brand_after'] = rows['first_other_brand_view_after'].notna().astype(int)
rival_before_buy = rows['first_other_brand_view_after'].notna() & (
    rows['first_purchase_after_view'].isna() | (rows['first_other_brand_view_after'] < rows['first_purchase_after_view']))
rows['next_step'] = np.select(
    [(rows['purchased'] == 1) & ~rival_before_buy, (rows['purchased'] == 1) & rival_before_buy,
     rows['bought_other_brand_after'] == 1, rows['bought_other_model_same_brand_after'] == 1,
     rows['viewed_other_brand_after'] == 1],
    ['bought_it', 'viewed_other_brand_then_bought_it', 'bought_other_brand',
     'bought_other_model_same_brand', 'viewed_other_brand_bought_nothing'], 'left_without_other_brand_view')

# ---------------------------------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------------------------------
cols = ['user_id', 'session_id', 'product_id', 'brand', 'brand_group', 'first_view', 'month', 'november',
        'initial_price', 'log_price', 'purchased', 'carted', 'purchased_after_cart',
        'first_cart_after_view', 'first_purchase_after_view',
        'earlier_tvs_in_visit', 'first_tv_in_visit', 'other_brand_before_in_visit', 'similar_rival_before_in_visit',
        'position_in_visit', 'prior_sessions', 'tv_purchase_before',
        'other_brand_before_48h', 'other_brand_before_7d', 'other_brand_before_any',
        'window_48h_complete', 'window_7d_complete',
        'viewed_other_brand_after', 'bought_other_brand_after', 'other_brand_bought_group',
        'bought_other_model_same_brand_after', 'next_step']
out = rows.sort_values(['first_view', 'user_id', 'session_id', 'product_id'], kind='stable')[cols]
assert not out.duplicated(KEY).any()
assert (out['purchased_after_cart'] <= out['carted']).all()
out.to_csv(DATA_DIR / 'tv_considerations.csv', index=False)

print(f'tv_events_clean.csv   {len(tv_ev):,} events ({len(raw) - len(ev):,} exact duplicates removed from the full file)')
print(f'tv_considerations.csv {len(out):,} rows, {out["user_id"].nunique():,} users')
print(out.groupby('brand_group').agg(rows=('purchased', 'size'), purchases=('purchased', 'sum'),
                                     conversion_pct=('purchased', lambda x: round(100 * x.mean(), 2))))
