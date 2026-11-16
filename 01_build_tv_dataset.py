# Builds the working datasets for the LG TV project from the original behaviour file, together with the tables that
# document and justify the build decisions.
# Run from Marketing-Project: python 01_build_tv_dataset.py
# Reads  ../data/01_LG_Electronics_Appliances/electronics_market_behavior.csv
# Writes ../data/01_LG_Electronics_Appliances/
#          tv_events_clean.csv                       event level, TV only, six brands
#          tv_considerations.csv                     one row per user x session x TV model
#          tv_considerations_top_models.csv          rows of LG's and Samsung's top models only
#          tv_considerations_without_top_models.csv  every other row (the other LG and Samsung models and all other_four rows)
#        results/build/
#          build_parameters.csv       the parameters used below, read by 02_preliminary_analysis.py
#          data_audit.csv             exclusions, key uniqueness, event-order checks and long sessions
#          top_model_selection.csv    every LG and Samsung model ranked by considerations, with the top-model rule applied
#          top_model_cutoffs.csv      the top-model split under the chosen rule and four alternatives
#          variable_decisions.csv     conversion by each variable kept in the file and by each browsing variant left out
#          reset_sensitivity.csv      position in the session with and without restarting the history after a purchase
# Column definitions are in tv_considerations_dictionary.md. Requires pandas and numpy.
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path('..') / 'data' / '01_LG_Electronics_Appliances'
OUT_DIR = Path('results') / 'build'
OUT_DIR.mkdir(parents=True, exist_ok=True)
KEY = ['user_id', 'session_id', 'product_id']
SESSION = ['user_id', 'session_id']
TOP_SHARE = 4.0                 # a model is a top model when it holds at least this % of its brand's considerations
PRICE_LOW, PRICE_HIGH = 0.8, 1.2  # price-similarity check: rival price between 80% and 120% of this model's price
HISTORY_DAYS = 7                # earlier-session checks use considerations at least this many days after the data start
GROUP_NAMES = {'lg': 'lg', 'samsung': 'samsung'}   # every other brand goes to 'other_four'
RESULT_NAMES = {'lg': 'bought_lg', 'samsung': 'bought_samsung', 'other_four': 'bought_other_four'}


def brand_group(brand):
    return pd.Series(np.where(brand.isin(['lg', 'samsung']), brand, 'other_four'), index=brand.index)


def position(n_before, other_before):
    return np.select([n_before == 0, other_before == 0], ['first_tv', 'after_same_brand_only'], 'after_other_brand')


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

# ---------------------------------------------------------------------------------------------
# 3. Outcome and funnel: this model, same session, at or after the first view. A cart or purchase logged in the same
#    second as the first view belongs to the consideration (counted in data_audit.csv).
# ---------------------------------------------------------------------------------------------
own_p = P.merge(rows[KEY + ['rid', 'first_view']], on=KEY).query('event_time >= first_view')
own_c = C.merge(rows[KEY + ['rid', 'first_view']], on=KEY).query('event_time >= first_view')
rows['purchased'] = rows['rid'].isin(own_p['rid']).astype(int)
rows['added_to_cart'] = rows['rid'].isin(own_c['rid']).astype(int)
rows['purchased_after_cart'] = (rows['rid'].map(own_p.groupby('rid')['event_time'].max())
                                > rows['rid'].map(own_c.groupby('rid')['event_time'].min())).astype(int)

# ---------------------------------------------------------------------------------------------
# 4. Earlier in the same session: TVs opened before this model, over the whole session. The version that restarts the
#    history after a TV purchase earlier in the session is built only for reset_sensitivity.csv.
# ---------------------------------------------------------------------------------------------
reset = (P[SESSION + ['event_time']].merge(rows[SESSION + ['rid', 'first_view']], on=SESSION)
         .query('event_time < first_view').groupby('rid')['event_time'].max())
rows['reset'] = rows['rid'].map(reset)
sv = rows[['rid'] + SESSION + ['brand', 'first_view', 'reset']].merge(
    V[SESSION + ['product_id', 'brand', 'event_time']].rename(
        columns={'product_id': 'v_product', 'brand': 'v_brand', 'event_time': 'v_time'}), on=SESSION)
sv = sv[sv['v_time'] < sv['first_view']]
sv['v_group'] = brand_group(sv['v_brand'])
rows['tvs_opened_before'] = rows['rid'].map(sv.groupby('rid')['v_product'].nunique()).fillna(0).astype(int)
for g in ['lg', 'samsung', 'other_four']:
    rows[f'{g}_opened_before'] = rows['rid'].isin(sv.loc[sv['v_group'] == g, 'rid']).astype(int)
rows['other_brand_opened_before'] = rows['rid'].isin(sv.loc[sv['v_brand'] != sv['brand'], 'rid']).astype(int)
rows['position_in_session'] = position(rows['tvs_opened_before'], rows['other_brand_opened_before'])

after_reset = sv[sv['reset'].isna() | (sv['v_time'] > sv['reset'])]
n_reset = rows['rid'].map(after_reset.groupby('rid')['v_product'].nunique()).fillna(0).astype(int)
other_reset = rows['rid'].isin(after_reset.loc[after_reset['v_brand'] != after_reset['brand'], 'rid']).astype(int)
rows['position_reset'] = position(n_reset, other_reset)

# ---------------------------------------------------------------------------------------------
# 5. Later in the same session: what was bought after this model was opened. One label per row, by priority (this
#    model, then the first other brand bought, then another model of the same brand), so it is a classification of
#    the consideration and does not list every purchase in the session.
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
# 6. Two flags from outside the session. The course file is a curated sample, so earlier sessions of a user can be
#    missing and both flags can only miss earlier visits and purchases, never invent them.
# ---------------------------------------------------------------------------------------------
up = rows[['rid', 'user_id', 'first_view']].merge(P[['user_id', 'event_time']], on='user_id')
rows['bought_tv_before'] = rows['rid'].isin(up.loc[up['event_time'] < up['first_view'], 'rid']).astype(int)
ss = ev.groupby(SESSION)['event_time'].agg(start='min', end='max').reset_index().sort_values('start', kind='stable')
ss['returning_visitor'] = (ss.groupby('user_id').cumcount() > 0).astype(int)
rows = rows.merge(ss[SESSION + ['returning_visitor']], on=SESSION, how='left')

# ---------------------------------------------------------------------------------------------
# 7. Top models: the LG and Samsung models that each hold at least TOP_SHARE % of their brand's considerations
# ---------------------------------------------------------------------------------------------
ms = (rows[rows['brand_group'].isin(['lg', 'samsung'])].groupby(['brand', 'product_id'])
      .agg(considerations=('rid', 'size'), purchased=('purchased', 'sum'), median_price=('price', 'median')).reset_index()
      .sort_values(['brand', 'considerations'], ascending=[True, False], kind='stable'))
ms['rank_in_brand'] = ms.groupby('brand').cumcount() + 1
ms['share_of_brand_pct'] = 100 * ms['considerations'] / ms.groupby('brand')['considerations'].transform('sum')
ms['cumulative_share_pct'] = ms.groupby('brand')['share_of_brand_pct'].cumsum()
ms['conversion_pct'] = 100 * ms['purchased'] / ms['considerations']
ms['top_model'] = (ms['share_of_brand_pct'] >= TOP_SHARE).astype(int)
ms.round(3).to_csv(OUT_DIR / 'top_model_selection.csv', index=False)
rows['top_model'] = rows['product_id'].isin(ms.loc[ms['top_model'] == 1, 'product_id']).astype(int)

RULES = [('Each model holds at least 3% of its brand', lambda d: d['share_of_brand_pct'] >= 3),
         (f'Each model holds at least {TOP_SHARE:g}% of its brand (chosen rule)', lambda d: d['share_of_brand_pct'] >= TOP_SHARE),
         ('Each model holds at least 5% of its brand', lambda d: d['share_of_brand_pct'] >= 5),
         ('Five most-considered models of each brand', lambda d: d['rank_in_brand'] <= 5),
         ('Ten most-considered models of each brand', lambda d: d['rank_in_brand'] <= 10)]
chosen = {br: set(ms.loc[(ms['brand'] == br) & (ms['top_model'] == 1), 'product_id']) for br in ['lg', 'samsung']}
cut = []
for label, rule in RULES:
    r = {'Rule': label}
    for br, name in [('lg', 'LG'), ('samsung', 'Samsung')]:
        d = ms[ms['brand'] == br]
        sel = rule(d)
        top, rest = d[sel], d[~sel]
        r[f'{name} top models'] = int(sel.sum())
        r[f'{name} same models as the chosen rule'] = 'yes' if set(top['product_id']) == chosen[br] else 'no'
        r[f'{name} share of considerations in top models (%)'] = top['share_of_brand_pct'].sum()
        r[f'{name} smallest share among top models (%)'] = top['share_of_brand_pct'].min()
        r[f'{name} largest share among other models (%)'] = rest['share_of_brand_pct'].max()
        r[f'{name} conversion in top models (%)'] = 100 * top['purchased'].sum() / top['considerations'].sum()
        r[f'{name} conversion in other models (%)'] = 100 * rest['purchased'].sum() / rest['considerations'].sum()
    r['LG minus Samsung conversion in top models (pp)'] = r['LG conversion in top models (%)'] - r['Samsung conversion in top models (%)']
    r['LG minus Samsung conversion in other models (pp)'] = r['LG conversion in other models (%)'] - r['Samsung conversion in other models (%)']
    cut.append(r)
cutoffs = pd.DataFrame(cut).set_index('Rule')
cutoffs.round(2).to_csv(OUT_DIR / 'top_model_cutoffs.csv')

# ---------------------------------------------------------------------------------------------
# 8. Variable checks (LG and Samsung considerations). The browsing variants that were left out are built here only for
#    this table and are not saved in the dataset. They can be rebuilt for a sensitivity analysis from tv_events_clean.csv.
# ---------------------------------------------------------------------------------------------
hist = rows[['rid', 'user_id', 'session_id', 'brand', 'first_view', 'price']].merge(
    V[['user_id', 'session_id', 'brand', 'event_time', 'price']].rename(
        columns={'session_id': 'v_session', 'brand': 'v_brand', 'event_time': 'v_time', 'price': 'v_price'}), on='user_id')
hist = hist[(hist['v_time'] < hist['first_view']) & (hist['v_brand'] != hist['brand'])]
earlier = hist[hist['v_session'] != hist['session_id']]
age = earlier['first_view'] - earlier['v_time']
same = hist[hist['v_session'] == hist['session_id']]
ratio = same['v_price'] / same['price']
similar_rids = set(same.loc[(ratio >= PRICE_LOW) & (ratio <= PRICE_HIGH), 'rid'])

lsr = rows[rows['brand_group'].isin(['lg', 'samsung'])]
history_start = lsr['first_view'].min().normalize() + pd.Timedelta(days=HISTORY_DAYS)
late = lsr[(lsr['first_view'] >= history_start) & (lsr['bought_tv_before'] == 0)]   # repeat buyers would mix two effects
not_in_session = late[late['other_brand_opened_before'] == 0]
CHECKS = [  # name, decision, comparison rows, reference rows, reference description
    ('Another brand opened earlier in the same session', 'kept',
     lsr[lsr['other_brand_opened_before'] == 1], lsr[lsr['other_brand_opened_before'] == 0], 'no other brand opened earlier in the session')]
for label, mask in [('within 48 hours', age <= pd.Timedelta(hours=48)), ('within 7 days', age <= pd.Timedelta(days=7)),
                    ('at any earlier time in the data', age.notna())]:
    ids = set(earlier.loc[mask, 'rid'])
    CHECKS.append((f'Another brand opened in an earlier session {label}, none earlier in this session', 'left out',
                   not_in_session[not_in_session['rid'].isin(ids)], not_in_session[~not_in_session['rid'].isin(ids)],
                   f'no other brand opened in this session or in that window (considerations from {history_start:%d %B}, '
                   'users with no earlier TV purchase)'))
with_rival = lsr[lsr['other_brand_opened_before'] == 1]
CHECKS += [
    (f'Other brand opened earlier in the session was priced at {100 * PRICE_LOW:.0f}% to {100 * PRICE_HIGH:.0f}% of this model', 'left out',
     with_rival[with_rival['rid'].isin(similar_rids)], with_rival[~with_rival['rid'].isin(similar_rids)],
     'other brand opened earlier in the session at a price outside that range'),
    ('Returning visitor (an earlier session in the data, any category)', 'kept',
     lsr[lsr['returning_visitor'] == 1], lsr[lsr['returning_visitor'] == 0], 'no earlier session in the data'),
    ('User bought a TV before this consideration (any session in the data)', 'kept',
     lsr[lsr['bought_tv_before'] == 1], lsr[lsr['bought_tv_before'] == 0], 'no earlier TV purchase in the data')]
dec = []
for name, decision, grp, ref, ref_text in CHECKS:
    r = {'Variable': name, 'Decision': decision, 'Reference group': ref_text}
    for br, nm in [('lg', 'LG'), ('samsung', 'Samsung')]:
        g_, f_ = grp[grp['brand'] == br], ref[ref['brand'] == br]
        r[f'{nm} considerations in the group'] = len(g_)
        r[f'{nm} conversion in the group (%)'] = 100 * g_['purchased'].mean()
        r[f'{nm} conversion in the reference group (%)'] = 100 * f_['purchased'].mean()
        r[f'{nm} difference (pp)'] = r[f'{nm} conversion in the group (%)'] - r[f'{nm} conversion in the reference group (%)']
    dec.append(r)
decisions = pd.DataFrame(dec).set_index('Variable')
decisions.round(2).to_csv(OUT_DIR / 'variable_decisions.csv')

# reset sensitivity: position over the whole session (used in the dataset) against position since the last TV purchase
rs = []
for definition, col in [('Whole session (used in the dataset)', 'position_in_session'), ('Since the last TV purchase in the session', 'position_reset')]:
    for pos in ['first_tv', 'after_same_brand_only', 'after_other_brand']:
        r = {'Definition': definition, 'Position': pos}
        for g in ['lg', 'samsung', 'other_four']:
            d = rows[(rows['brand_group'] == g) & (rows[col] == pos)]
            r[f'{g} considerations'] = len(d)
            r[f'{g} conversion (%)'] = 100 * d['purchased'].mean()
        rs.append(r)
reset_table = pd.DataFrame(rs).set_index(['Definition', 'Position'])
reset_table.round(2).to_csv(OUT_DIR / 'reset_sensitivity.csv')

# ---------------------------------------------------------------------------------------------
# 9. Data audit
# ---------------------------------------------------------------------------------------------
lo_t, hi_t = pd.Timestamp('2019-10-01', tz='UTC'), pd.Timestamp('2019-12-01', tz='UTC')
tv_views = V.groupby(KEY)['event_time'].min().rename('first_view_any')
pa = P.join(tv_views, on=KEY)
ca = C.join(tv_views, on=KEY)
ss['hours'] = (ss['end'] - ss['start']).dt.total_seconds() / 3600
tv_sessions = rows[SESSION].drop_duplicates().merge(ss, on=SESSION)
per_session = rows.groupby(SESSION).size()
audit = [
    ('Raw events in the original file', len(raw), ''),
    ('Exact duplicate events removed', len(raw) - len(ev), 'identical on all eight columns'),
    ('Events in other product categories, left out of the TV files', int((ev['product_category'] != 'tv').sum()), ''),
    ('TV events kept', len(tv_ev), ', '.join(f'{k} {v:,}' for k, v in tv_ev['event_type'].value_counts().items())),
    ('Events outside 1 October to 30 November 2019', int(((ev['event_time'] < lo_t) | (ev['event_time'] >= hi_t)).sum()), 'all categories'),
    ('Session IDs that appear under more than one user', int((ev.groupby('session_id')['user_id'].nunique() > 1).sum()),
     'sessions are identified by user and session ID together'),
    ('Duplicate user x session x model keys in tv_considerations.csv', int(rows.duplicated(KEY).sum()), ''),
    ('TV purchase events with no view of the model in the same session', int(pa['first_view_any'].isna().sum()), ''),
    ('TV purchase events logged before the first view of the model in the session', int((pa['event_time'] < pa['first_view_any']).sum()), ''),
    ('TV cart events with no view of the model in the same session', int(ca['first_view_any'].isna().sum()), ''),
    ('TV cart events logged before the first view of the model in the session', int((ca['event_time'] < ca['first_view_any']).sum()), ''),
    ('Cart events logged in the same second as the first view', int((own_c['event_time'] == own_c['first_view']).sum()), 'counted as part of the consideration'),
    ('Purchase events logged in the same second as the first view', int((own_p['event_time'] == own_p['first_view']).sum()), ''),
    ('Purchased considerations with a cart event but no purchase after it', int(((rows['purchased'] == 1) & (rows['added_to_cart'] == 1) & (rows['purchased_after_cart'] == 0)).sum()), ''),
    ('Considerations opened after a TV purchase earlier in the same session', int(rows['reset'].notna().sum()), 'see reset_sensitivity.csv'),
    ('TV sessions lasting more than 24 hours, first to last event in any category', int((tv_sessions['hours'] > 24).sum()), ''),
    ('Longest TV session, first to last event (hours)', round(float(tv_sessions['hours'].max()), 1), ''),
    ('99th percentile of TV session length (hours)', round(float(tv_sessions['hours'].quantile(0.99)), 2), ''),
    ('Most TV models opened in one session', int(per_session.max()), ''),
    ('Sessions with 20 or more TV models opened', int((per_session >= 20).sum()),
     f'{100 * per_session[per_session >= 20].sum() / per_session.sum():.1f}% of considerations'),
    ('Users with 100 or more TV considerations', int((rows.groupby('user_id').size() >= 100).sum()), ''),
]
audit = pd.DataFrame(audit, columns=['Check', 'Value', 'Note']).set_index('Check')
audit.to_csv(OUT_DIR / 'data_audit.csv')

pd.Series({'top_model_share_pct': TOP_SHARE, 'price_similarity_low': PRICE_LOW, 'price_similarity_high': PRICE_HIGH,
           'history_days': HISTORY_DAYS, 'history_check_start': f'{history_start:%Y-%m-%d}'}, name='value').rename_axis(
    'parameter').to_csv(OUT_DIR / 'build_parameters.csv')

# ---------------------------------------------------------------------------------------------
# 10. Save
# ---------------------------------------------------------------------------------------------
cols = ['user_id', 'session_id', 'product_id', 'brand', 'brand_group', 'top_model', 'first_view', 'month', 'price',
        'purchased', 'added_to_cart', 'purchased_after_cart',
        'tvs_opened_before', 'lg_opened_before', 'samsung_opened_before', 'other_four_opened_before',
        'other_brand_opened_before', 'position_in_session', 'session_result',
        'bought_tv_before', 'returning_visitor']
out = rows.sort_values(['first_view', 'user_id', 'session_id', 'product_id'], kind='stable')[cols]
assert not out.duplicated(KEY).any()
assert (out['purchased_after_cart'] <= out['added_to_cart']).all()
assert ((out['session_result'] == 'bought_this_model') == (out['purchased'] == 1)).all()
assert (out.loc[out['brand_group'] == 'other_four', 'top_model'] == 0).all()
assert (out[out['position_in_session'] == 'first_tv'].groupby(SESSION).size() == 1).all()
top_rows, other_rows = out[out['top_model'] == 1], out[out['top_model'] == 0]
assert len(top_rows) + len(other_rows) == len(out)
out.to_csv(DATA_DIR / 'tv_considerations.csv', index=False)
top_rows.to_csv(DATA_DIR / 'tv_considerations_top_models.csv', index=False)
other_rows.to_csv(DATA_DIR / 'tv_considerations_without_top_models.csv', index=False)

pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 30)
print(f'tv_events_clean.csv                       {len(tv_ev):,} events')
print(f'tv_considerations.csv                     {len(out):,} rows, {out["user_id"].nunique():,} users')
print(f'tv_considerations_top_models.csv          {len(top_rows):,} rows')
print(f'tv_considerations_without_top_models.csv  {len(other_rows):,} rows')
print(out.groupby('brand_group').agg(rows=('purchased', 'size'), purchases=('purchased', 'sum'),
                                     conversion_pct=('purchased', lambda x: round(100 * x.mean(), 2))))
n_top = ms.groupby('brand')['top_model'].sum()
print(f'\nTop-model rule: a model is a top model when it holds at least {TOP_SHARE:g}% of its brand\'s considerations. '
      f'No LG or Samsung model holds between {ms.loc[ms["top_model"] == 0, "share_of_brand_pct"].max():.2f}% and '
      f'{ms.loc[ms["top_model"] == 1, "share_of_brand_pct"].min():.2f}%, so any cutoff in that range gives the same split. '
      f'Top models: LG {n_top["lg"]}, Samsung {n_top["samsung"]}.')
print(f'\nData audit (results/build/data_audit.csv)\n{audit.to_string()}')
print(f'\nThe top-model split under five rules (results/build/top_model_cutoffs.csv)\n{cutoffs.round(2).T.to_string()}')
print(f'\nVariable checks (results/build/variable_decisions.csv)\n{decisions.drop(columns="Reference group").round(2).to_string()}')
print(f'\nReset sensitivity (results/build/reset_sensitivity.csv)\n{reset_table.round(2).to_string()}')
