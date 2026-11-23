# Builds the two working files of the LG TV project from the course behaviour file and checks every derived column
# with independent code. Columns are defined in tv_considerations_dictionary.md.
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path('..') / 'data' / '01_LG_Electronics_Appliances'
KEY = ['user_id', 'session_id', 'product_id']
SESSION = ['user_id', 'session_id']
TOP_SHARE = 4.0      # a top model holds at least this % of its brand's considerations over the two months
SEED = 2026          # seed of the sample for the independent check in section 9
RESULT_NAMES = {'lg': 'bought_lg', 'samsung': 'bought_samsung', 'other_four': 'bought_other_four'}
pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 30)


def brand_group(brand):
    return pd.Series(np.where(brand.isin(['lg', 'samsung']), brand, 'other_four'), index=brand.index)


def position(n_before, other_before):
    return np.select([n_before == 0, other_before == 0], ['first_tv', 'after_same_brand_only'], 'after_other_brand')


# ---------------------------------------------------------------------------------------------
# 1. Clean events: drop exact duplicates, parse times, standardise text, keep all six categories
# ---------------------------------------------------------------------------------------------
raw = pd.read_csv(DATA_DIR / 'electronics_market_behavior.csv')
ev = raw.drop_duplicates().copy()
ev['event_time'] = pd.to_datetime(ev['event_time'], utc=True)
for col in ['event_type', 'brand', 'product_category']:
    ev[col] = ev[col].str.strip().str.lower()
assert ev.notna().all().all() and (ev['price'] > 0).all()
assert ev['event_type'].isin(['view', 'cart', 'purchase']).all()
assert (ev.groupby('product_id')[['brand', 'product_category']].nunique() == 1).all().all()
ev = ev.sort_values(['event_time', 'user_id', 'session_id'], kind='stable').reset_index(drop=True)

tv_ev = ev[ev['product_category'] == 'tv']
V = tv_ev[tv_ev['event_type'] == 'view'][KEY + ['brand', 'event_time', 'price']]
C = tv_ev[tv_ev['event_type'] == 'cart'][KEY + ['event_time']]
P = tv_ev[tv_ev['event_type'] == 'purchase'][KEY + ['brand', 'event_time']]

# ---------------------------------------------------------------------------------------------
# 2. One row per user x session x TV model, timed at the first view of that model in that session
# ---------------------------------------------------------------------------------------------
rows = (V.sort_values('event_time', kind='stable').drop_duplicates(KEY)
        .rename(columns={'event_time': 'first_view'}).reset_index(drop=True))   # price is the price at the first view
rows['rid'] = np.arange(len(rows))
rows['brand_group'] = brand_group(rows['brand'])
rows['month'] = rows['first_view'].dt.strftime('%Y-%m')
rows['week'] = rows['first_view'].dt.isocalendar().week.astype(int)

# ---------------------------------------------------------------------------------------------
# 3. Outcome and funnel: this model, same session, at or after the first view
# ---------------------------------------------------------------------------------------------
own_p = P.merge(rows[KEY + ['rid', 'first_view']], on=KEY).query('event_time >= first_view')
own_c = C.merge(rows[KEY + ['rid', 'first_view']], on=KEY).query('event_time >= first_view')
rows['purchased'] = rows['rid'].isin(own_p['rid']).astype(int)
rows['added_to_cart'] = rows['rid'].isin(own_c['rid']).astype(int)
rows['purchased_after_cart'] = (rows['rid'].map(own_p.groupby('rid')['event_time'].max())
                                > rows['rid'].map(own_c.groupby('rid')['event_time'].min())).astype(int)

# ---------------------------------------------------------------------------------------------
# 4. Earlier in the same session: TVs opened before this model, over the whole session
# ---------------------------------------------------------------------------------------------
sv = rows[['rid'] + SESSION + ['brand', 'first_view']].merge(
    V[SESSION + ['product_id', 'brand', 'event_time']].rename(
        columns={'product_id': 'v_product', 'brand': 'v_brand', 'event_time': 'v_time'}), on=SESSION)
sv = sv[sv['v_time'] < sv['first_view']]
sv['v_group'] = brand_group(sv['v_brand'])
rows['tvs_opened_before'] = rows['rid'].map(sv.groupby('rid')['v_product'].nunique()).fillna(0).astype(int)
for g in ['lg', 'samsung', 'other_four']:
    rows[f'{g}_opened_before'] = rows['rid'].isin(sv.loc[sv['v_group'] == g, 'rid']).astype(int)
rows['other_brand_opened_before'] = rows['rid'].isin(sv.loc[sv['v_brand'] != sv['brand'], 'rid']).astype(int)
rows['position_in_session'] = position(rows['tvs_opened_before'], rows['other_brand_opened_before'])
# The same history in the same terms for both main brands. Empty for the other four brands.
is_lg, is_sam = rows['brand_group'] == 'lg', rows['brand_group'] == 'samsung'
rows['rival_opened_before'] = np.select([is_lg, is_sam], [rows['samsung_opened_before'], rows['lg_opened_before']], np.nan)
rows['own_other_model_opened_before'] = np.select([is_lg, is_sam], [rows['lg_opened_before'], rows['samsung_opened_before']], np.nan)

# ---------------------------------------------------------------------------------------------
# 5. Later in the same session: what was bought after this model was opened. One label per row, by priority (this
#    model, then the first other brand bought, then another model of the same brand).
# ---------------------------------------------------------------------------------------------
sp = rows[['rid'] + KEY + ['brand', 'first_view']].merge(
    P[SESSION + ['product_id', 'brand', 'event_time']].rename(
        columns={'product_id': 'p_product', 'brand': 'p_brand', 'event_time': 'p_time'}), on=SESSION)
sp = sp[sp['p_time'] > sp['first_view']]
other = sp[sp['p_brand'] != sp['brand']].sort_values('p_time', kind='stable').drop_duplicates('rid')
other_result = rows['rid'].map(other.set_index('rid')['p_brand'].pipe(brand_group).map(RESULT_NAMES))
same_brand_other_model = sp.loc[(sp['p_brand'] == sp['brand']) & (sp['p_product'] != sp['product_id']), 'rid']
rows['session_result'] = np.select(
    [rows['purchased'] == 1, other_result.notna(), rows['rid'].isin(same_brand_other_model)],
    ['bought_this_model', other_result.fillna(''), 'bought_other_model_same_brand'], 'bought_nothing')

# ---------------------------------------------------------------------------------------------
# 6. The user's earlier activity: a TV bought before the first view (earlier in this session or in an earlier session)
#    and an earlier session in any category
# ---------------------------------------------------------------------------------------------
up = rows[['rid', 'user_id', 'first_view']].merge(P[['user_id', 'event_time']], on='user_id')
rows['bought_tv_before'] = rows['rid'].isin(up.loc[up['event_time'] < up['first_view'], 'rid']).astype(int)
ss = ev.groupby(SESSION)['event_time'].min().rename('start').reset_index().sort_values('start', kind='stable')
ss['returning_visitor'] = (ss.groupby('user_id').cumcount() > 0).astype(int)
rows = rows.merge(ss[SESSION + ['returning_visitor']], on=SESSION, how='left')

# ---------------------------------------------------------------------------------------------
# 7. Each LG and Samsung model's share of its brand's considerations over the two months, its rank and the top-model flag
# ---------------------------------------------------------------------------------------------
ms = (rows[rows['brand_group'].isin(['lg', 'samsung'])].groupby(['brand', 'product_id'])
      .agg(considerations=('rid', 'size'), purchased=('purchased', 'sum')).reset_index()
      .sort_values(['brand', 'considerations'], ascending=[True, False], kind='stable'))
ms['model_rank_in_brand'] = ms.groupby('brand').cumcount() + 1   # ties are ordered by product ID
ms['model_share_of_brand_pct'] = 100 * ms['considerations'] / ms.groupby('brand')['considerations'].transform('sum')
rows = rows.merge(ms[['product_id', 'model_share_of_brand_pct', 'model_rank_in_brand']], on='product_id', how='left')
rows['top_model'] = (rows['model_share_of_brand_pct'] >= TOP_SHARE).astype(int)

gap_low = ms.loc[ms['model_share_of_brand_pct'] < TOP_SHARE, 'model_share_of_brand_pct'].max()
gap_high = ms.loc[ms['model_share_of_brand_pct'] >= TOP_SHARE, 'model_share_of_brand_pct'].min()

# ---------------------------------------------------------------------------------------------
# 8. Checks on the full dataset
# ---------------------------------------------------------------------------------------------
COLS = ['user_id', 'session_id', 'product_id', 'brand', 'brand_group', 'first_view', 'month', 'week', 'price',
        'purchased', 'added_to_cart', 'purchased_after_cart',
        'tvs_opened_before', 'lg_opened_before', 'samsung_opened_before', 'other_four_opened_before',
        'other_brand_opened_before', 'rival_opened_before', 'own_other_model_opened_before', 'position_in_session',
        'session_result', 'bought_tv_before', 'returning_visitor',
        'model_share_of_brand_pct', 'model_rank_in_brand', 'top_model']
out = rows.sort_values(['first_view', 'user_id', 'session_id', 'product_id'], kind='stable')[COLS].reset_index(drop=True)
main = out['brand_group'] != 'other_four'
assert not out.duplicated(KEY).any()
assert (out['purchased_after_cart'] <= out['added_to_cart']).all()
assert ((out['session_result'] == 'bought_this_model') == (out['purchased'] == 1)).all()
assert (out[out['position_in_session'] == 'first_tv'].groupby(SESSION).size() == 1).all()
assert (out.loc[~main, 'top_model'] == 0).all() and out.loc[~main, ['rival_opened_before', 'model_share_of_brand_pct']].isna().all().all()
assert out.loc[main, ['rival_opened_before', 'own_other_model_opened_before', 'model_share_of_brand_pct']].notna().all().all()
assert (out.groupby('product_id')['top_model'].nunique() == 1).all()
assert gap_low < TOP_SHARE <= gap_high

# ---------------------------------------------------------------------------------------------
# 9. Independent check. Recomputes every derived column for a stratified sample of rows with plain loops over each
#    user's events from the original file, sharing no code with sections 2 to 7, and stops if any value disagrees.
#    Logical checks then run on every row.
# ---------------------------------------------------------------------------------------------
group = lambda b: b if b in ('lg', 'samsung') else 'other_four'
rng = np.random.default_rng(SEED)
strata = {'random': out.index,
          'purchased': out.index[out['purchased'] == 1],
          'bought_tv_before': out.index[out['bought_tv_before'] == 1],
          'other_brand_opened_before': out.index[out['other_brand_opened_before'] == 1],
          'another model or brand bought': out.index[~out['session_result'].isin(['bought_nothing', 'bought_this_model'])],
          'three or more TVs opened before': out.index[out['tvs_opened_before'] >= 3],
          'top model': out.index[out['top_model'] == 1]}
idx = np.unique(np.concatenate([rng.choice(v, min(len(v), 1500 if k == 'random' else 400), replace=False)
                                for k, v in strata.items()]))
sample = out.loc[idx]
users = set(sample['user_id'])
tv_raw = raw[raw['product_category'].str.strip().str.lower() == 'tv'].drop_duplicates().copy()
tv_raw['event_time'] = pd.to_datetime(tv_raw['event_time'], utc=True)
uev = {u: g.sort_values('event_time', kind='stable') for u, g in tv_raw[tv_raw['user_id'].isin(users)].groupby('user_id')}
all_raw = raw[raw['user_id'].isin(users)].drop_duplicates().copy()
all_raw['event_time'] = pd.to_datetime(all_raw['event_time'], utc=True)
starts = all_raw.groupby(['user_id', 'session_id'])['event_time'].min()
tv_views = tv_raw[tv_raw['event_type'].str.strip().str.lower() == 'view'].drop_duplicates(KEY)
brand_counts = {}
for (b_, pid), n in tv_views.groupby(['brand', 'product_id']).size().items():
    brand_counts.setdefault(b_, {})[pid] = n

CHECK = ['first_view', 'week', 'price', 'purchased', 'added_to_cart', 'purchased_after_cart', 'tvs_opened_before',
         'lg_opened_before', 'samsung_opened_before', 'other_four_opened_before', 'other_brand_opened_before',
         'rival_opened_before', 'own_other_model_opened_before', 'position_in_session', 'session_result',
         'bought_tv_before', 'returning_visitor', 'model_share_of_brand_pct', 'model_rank_in_brand', 'top_model']
agree = {k: [] for k in CHECK}
for _, r in sample.iterrows():
    E = uev[r['user_id']]
    S = E[E['session_id'] == r['session_id']]
    own = S[S['product_id'] == r['product_id']]
    t0 = own.loc[own['event_type'] == 'view', 'event_time'].min()
    p0 = own.loc[(own['event_type'] == 'view') & (own['event_time'] == t0), 'price'].iloc[0]
    buys = own[(own['event_type'] == 'purchase') & (own['event_time'] >= t0)]
    carts = own[(own['event_type'] == 'cart') & (own['event_time'] >= t0)]
    before = S[(S['event_type'] == 'view') & (S['event_time'] < t0)]
    groups_before = set(before['brand'].map(group))
    other_before = int((before['brand'] != r['brand']).any())
    n_before = before['product_id'].nunique()
    later = S[(S['event_type'] == 'purchase') & (S['event_time'] > t0)]
    other_buys = later[later['brand'] != r['brand']]
    same_brand_other = later[(later['brand'] == r['brand']) & (later['product_id'] != r['product_id'])]
    if len(buys):
        result = 'bought_this_model'
    elif len(other_buys):
        result = RESULT_NAMES[group(other_buys.iloc[0]['brand'])]
    elif len(same_brand_other):
        result = 'bought_other_model_same_brand'
    else:
        result = 'bought_nothing'
    g = group(r['brand'])
    rival = {'lg': 'samsung', 'samsung': 'lg'}.get(g)
    counts = brand_counts.get(r['brand'], {})
    total = sum(counts.values())
    share = 100 * counts[r['product_id']] / total if g != 'other_four' else np.nan
    own_n = counts.get(r['product_id'])
    rank = (1 + sum(1 for p_, n in counts.items() if n > own_n or (n == own_n and p_ < r['product_id']))
            if g != 'other_four' else np.nan)   # ties are ordered by product ID
    sess = starts.loc[r['user_id']]
    mine = {'first_view': t0, 'week': int(t0.isocalendar()[1]), 'price': p0, 'purchased': int(len(buys) > 0),
            'added_to_cart': int(len(carts) > 0),
            'purchased_after_cart': int(len(carts) > 0 and len(buys) > 0 and buys['event_time'].max() > carts['event_time'].min()),
            'tvs_opened_before': n_before, 'lg_opened_before': int('lg' in groups_before),
            'samsung_opened_before': int('samsung' in groups_before),
            'other_four_opened_before': int('other_four' in groups_before), 'other_brand_opened_before': other_before,
            'rival_opened_before': int(rival in groups_before) if rival else np.nan,
            'own_other_model_opened_before': int(g in groups_before) if rival else np.nan,
            'position_in_session': 'first_tv' if n_before == 0 else 'after_other_brand' if other_before else 'after_same_brand_only',
            'session_result': result,
            'bought_tv_before': int(((E['event_type'] == 'purchase') & (E['event_time'] < t0)).any()),
            'returning_visitor': int((sess < sess.loc[r['session_id']]).any()),
            'model_share_of_brand_pct': share, 'model_rank_in_brand': rank,
            'top_model': int(share >= TOP_SHARE) if g != 'other_four' else 0}
    for k, v in mine.items():
        if isinstance(v, float) and np.isnan(v):
            agree[k].append(pd.isna(r[k]))
        elif k in ('price', 'model_share_of_brand_pct'):
            agree[k].append(abs(v - r[k]) < 1e-9)
        else:
            agree[k].append(v == r[k])
rates_ = pd.Series({k: 100 * np.mean(v) for k, v in agree.items()}, name='Agreement (%)')
print(f'\nIndependent check on {len(sample):,} rows (stratified so that every special case is included)')
print(rates_.round(2).to_string())
assert (rates_ == 100).all(), 'A derived column disagrees with the independent recomputation'

logic = {
    'brand_group matches brand': (out['brand_group'] == out['brand'].map(group)).all(),
    'month and week match first_view': ((out['month'] == out['first_view'].dt.strftime('%Y-%m'))
                                        & (out['week'] == out['first_view'].dt.isocalendar().week.astype(int))).all(),
    'a brand flag is set only when a TV was opened before': (
        (out[['lg_opened_before', 'samsung_opened_before', 'other_four_opened_before']].max(axis=1) == 1)
        == (out['tvs_opened_before'] > 0)).all(),
    'position after_other_brand exactly when another brand was opened before': (
        (out['position_in_session'] == 'after_other_brand') == (out['other_brand_opened_before'] == 1)).all(),
    'LG and Samsung rows: rival flag equals another brand among the main two': (
        out.loc[main, 'rival_opened_before'] == np.where(out.loc[main, 'brand_group'] == 'lg', out.loc[main, 'samsung_opened_before'],
                                                         out.loc[main, 'lg_opened_before'])).all(),
    'an LG row cannot end as bought_lg, a Samsung row as bought_samsung': not (
        ((out['brand_group'] == 'lg') & (out['session_result'] == 'bought_lg'))
        | ((out['brand_group'] == 'samsung') & (out['session_result'] == 'bought_samsung'))).any(),
    'model shares add up to 100% within each brand': np.allclose(
        out[main].drop_duplicates('product_id').groupby('brand')['model_share_of_brand_pct'].sum(), 100),
}
print('\nLogical checks on all rows')
for k, v in logic.items():
    print(f'   {"PASS" if v else "FAIL"}  {k}')
assert all(logic.values())

# ---------------------------------------------------------------------------------------------
# 10. Save the two files, only after every check above has passed
# ---------------------------------------------------------------------------------------------
ev.to_csv(DATA_DIR / 'events_clean.csv', index=False)
out.to_csv(DATA_DIR / 'tv_considerations.csv', index=False)
print('\nWrote events_clean.csv and tv_considerations.csv')
