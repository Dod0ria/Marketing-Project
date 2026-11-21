# Preliminary analysis for report Sections 1 to 4 of the LG TV project. Descriptive only, no models.
# Run from Marketing-Project after 01_build_tv_dataset.py: python 02_preliminary_analysis.py
# Inputs, all in ../data/01_LG_Electronics_Appliances/ unless marked
#   tv_considerations.csv, tv_considerations_top_models.csv, tv_considerations_without_top_models.csv and
#   tv_events_clean.csv (built by 01_build_tv_dataset.py)
#   electronics_market_behavior.csv (original file, used for the sample counts and the six-category comparison)
#   appliance_brand_reviews.csv (original file, used only to record that it holds no TVs)
#   results/build/*.csv (the decision tables written by 01_build_tv_dataset.py)
# Outputs results/preliminary/tables/*.csv, results/preliminary/figures/*.png and results/preliminary/preliminary_analysis.html.
# Every table and figure states the file it is built from. Table K lists the figures quoted in the text that do not
# appear in another table, so that every number used in the report is computed here.
# A consideration is one row of tv_considerations.csv (a user, in one session, opening one TV model).
# Requires pandas, numpy, matplotlib and statsmodels.
from pathlib import Path
import html
import itertools
import textwrap
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

DATA_DIR = Path('..') / 'data' / '01_LG_Electronics_Appliances'
BUILD = Path('results') / 'build'
OUT = Path('results') / 'preliminary'
(OUT / 'tables').mkdir(parents=True, exist_ok=True)
(OUT / 'figures').mkdir(parents=True, exist_ok=True)

F_CONS, F_TOP, F_REST = 'tv_considerations.csv', 'tv_considerations_top_models.csv', 'tv_considerations_without_top_models.csv'
F_EV, F_RAW, F_REV = 'tv_events_clean.csv', 'electronics_market_behavior.csv', 'appliance_brand_reviews.csv'
F_SEL, F_CUT, F_DEC = 'results/build/top_model_selection.csv', 'results/build/top_model_cutoffs.csv', 'results/build/variable_decisions.csv'
F_AUD, F_RES, F_PAR = 'results/build/data_audit.csv', 'results/build/reset_sensitivity.csv', 'results/build/build_parameters.csv'

KEY = ['user_id', 'session_id', 'product_id']
SESSION = ['user_id', 'session_id']
NAMES = {'lg': 'LG', 'samsung': 'Samsung', 'artel': 'Artel', 'haier': 'Haier', 'kivi': 'Kivi', 'tcl': 'TCL',
         'other_four': 'Other four brands'}
CATEGORIES = {'tv': 'TV', 'washing_machine': 'Washing machine', 'refrigerator': 'Refrigerator',
              'vacuum_cleaner': 'Vacuum cleaner', 'microwave': 'Microwave', 'monitor': 'Monitor'}
COLORS = {'lg': '#A50034', 'samsung': '#1428A0', 'other_four': '#8A8D91', 'artel': '#8A8D91', 'haier': '#8A8D91',
          'kivi': '#8A8D91', 'tcl': '#8A8D91'}
GROUPS = ['lg', 'samsung', 'other_four']
POSITION = {'first_tv': 'First TV opened in the session', 'after_same_brand_only': 'Opened after other models of the same brand only',
            'after_other_brand': 'Opened after a TV of another brand'}
RESULT = {'bought_this_model': 'Bought this model', 'bought_other_model_same_brand': 'Bought another model of the same brand',
          'bought_lg': 'Bought an LG TV', 'bought_samsung': 'Bought a Samsung TV', 'bought_other_four': 'Bought a TV of the other four brands',
          'bought_nothing': 'Bought no TV'}
BANDS, BAND_NAMES = [0, 300, 450, 650, 1000, np.inf], ['Under 300', '300 to 450', '450 to 650', '650 to 1,000', 'Over 1,000']
UNIT = 'A consideration is a user, in one session, opening one TV model, timed at the first view.'
# Inference convention. Rates and gaps over considerations use standard errors clustered by user and by model (two-way).
# Rates over sessions use standard errors clustered by user. Other conventions appear only as labelled sensitivities.
CI_NOTE = 'Bars show 95% intervals with standard errors clustered by user and by model.'
CI_NOTE_SESSIONS = 'Bars show 95% intervals with standard errors clustered by user.'

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 9.5, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'axes.labelsize': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
    'grid.color': '#e4e4e4', 'grid.linewidth': 0.6, 'axes.axisbelow': True, 'legend.frameon': False,
    'legend.fontsize': 8.5, 'savefig.dpi': 200})
pct = FuncFormatter(lambda v, _: f'{v:g}%')
EXHIBITS, KEYFIG = [], []


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------
def built_from(src):
    return 'Built from ' + (src[0] if len(src) == 1 else ', '.join(src[:-1]) + ' and ' + src[-1]) + '.'


def mean_ci(y, *clusters):
    """Mean of y in % with a 95% interval whose standard error is clustered on one or two dimensions."""
    y = np.asarray(y, float)
    e, n = y - y.mean(), len(y)

    def v(groups):
        s = pd.Series(e).groupby(np.asarray(groups)).sum()
        return 0.0 if len(s) < 2 else len(s) / (len(s) - 1) * (s ** 2).sum() / n ** 2

    if len(clusters) == 1:
        var = v(clusters[0])
    else:
        a, b = np.asarray(clusters[0]), np.asarray(clusters[1])
        var = v(a) + v(b) - v(pd.Series(a).astype(str) + '|' + pd.Series(b).astype(str))
        var = max(var, v(a), v(b)) if var <= 0 else var
    se = np.sqrt(var)
    return 100 * y.mean(), 100 * (y.mean() - 1.96 * se), 100 * (y.mean() + 1.96 * se)


def rates(df, by, y='purchased'):
    """Rates over considerations with intervals clustered by user and by model."""
    out = {}
    for k, d in df.groupby(by, observed=True):
        m, lo_, hi_ = mean_ci(d[y], d['user_id'], d['product_id'])
        out[k] = {'rows': len(d), 'events': int(d[y].sum()), 'rate_pct': m, 'ci_low_pct': lo_, 'ci_high_pct': hi_}
    g = pd.DataFrame(out).T
    g.index.names = by if isinstance(by, list) else [by]
    return g.astype({'rows': int, 'events': int})


def wild_p(d, draws=9999, seed=2026):
    """Wild cluster bootstrap p-value for the LG minus Samsung gap, clusters = models, Rademacher weights, null imposed.
    With 16 or fewer models every sign pattern is enumerated."""
    g = d[d['brand'].isin(['lg', 'samsung'])].groupby('product_id').agg(
        n=('purchased', 'size'), s=('purchased', 'sum'), lg=('brand', lambda x: x.iloc[0] == 'lg'))
    n, s, lg = g['n'].to_numpy(float), g['s'].to_numpy(float), g['lg'].to_numpy(bool)
    NL, NS = n[lg].sum(), n[~lg].sum()

    def t_stat(S):
        S = np.atleast_2d(S)
        mL, mS = S[:, lg].sum(1) / NL, S[:, ~lg].sum(1) / NS
        var = ((S[:, lg] - np.outer(mL, n[lg])) ** 2).sum(1) / NL ** 2 + ((S[:, ~lg] - np.outer(mS, n[~lg])) ** 2).sum(1) / NS ** 2
        return (mL - mS) / np.sqrt(var)

    t0 = t_stat(s)[0]
    ybar = s.sum() / n.sum()
    W = (np.array(list(itertools.product([-1.0, 1.0], repeat=len(n)))) if len(n) <= 16
         else np.random.default_rng(seed).choice([-1.0, 1.0], size=(draws, len(n))))
    t_star = t_stat(n * ybar + W * (s - n * ybar))
    return float(np.mean(np.abs(t_star) >= abs(t0) - 1e-12)), len(n)


def save_table(tid, title, df, note, src, feeds=''):
    df = df.round(2)
    note = f'{note} {built_from(src)}'.strip()
    path = OUT / 'tables' / f"T{tid.replace('.', '_')}.csv"
    df.to_csv(path)
    EXHIBITS.append(dict(kind='Table', id=tid, title=title, note=note, src=src, feeds=feeds, df=df, path=path))
    print(f'\nTable {tid}. {title}\n{df.to_string()}')


def finish(fig, fid, title, note, src, feeds=''):
    note = f'{note} {built_from(src)}'.strip()
    w, _ = fig.get_size_inches()
    fig.suptitle(f'Figure {fid}. {title}', x=0.01, ha='left', fontsize=10.5, fontweight='bold')
    fig.text(0.01, -0.015, '\n'.join(textwrap.wrap('Note. ' + note, width=int(w * 17))), fontsize=7.2, color='#555555', ha='left', va='top')
    path = OUT / 'figures' / f"F{fid.replace('.', '_')}.png"
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    EXHIBITS.append(dict(kind='Figure', id=fid, title=title, note=note, src=src, feeds=feeds, path=path))
    print(f'\nFigure {fid}. {title} -> {path}')


def key_figure(topic, measure, value, src):
    KEYFIG.append({'Topic': topic, 'What is measured': measure, 'Value': value, 'Built from': ', '.join(src)})


def errbar_bars(ax, x, r, color, width, label=None):
    ax.bar(x, r['rate_pct'], width, color=color, label=label)
    ax.errorbar(x, r['rate_pct'], yerr=[r['rate_pct'] - r['ci_low_pct'], r['ci_high_pct'] - r['rate_pct']],
                fmt='none', ecolor='#333333', elinewidth=0.8, capsize=2.5)


def gap_precision(d):
    """LG minus Samsung conversion and its standard error under three assumptions, in percentage points. The main
    convention clusters by user and by model, the other two are sensitivities."""
    d = d[d['brand'].isin(['lg', 'samsung'])].assign(lg=lambda x: (x['brand'] == 'lg').astype(int))
    m = smf.ols('purchased ~ lg', d)
    se = {'independent': m.fit().bse['lg'],
          'product': m.fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d['product_id'])[0]}).bse['lg'],
          'user_product': m.fit(cov_type='cluster', cov_kwds={'groups': np.column_stack(
              [pd.factorize(d['user_id'])[0], pd.factorize(d['product_id'])[0]])}).bse['lg']}
    gap = d.loc[d['lg'] == 1, 'purchased'].mean() - d.loc[d['lg'] == 0, 'purchased'].mean()
    return 100 * gap, {k: 100 * v for k, v in se.items()}


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------
cons = pd.read_csv(DATA_DIR / F_CONS, parse_dates=['first_view'])
top = pd.read_csv(DATA_DIR / F_TOP, parse_dates=['first_view'])
rest = pd.read_csv(DATA_DIR / F_REST, parse_dates=['first_view'])
events = pd.read_csv(DATA_DIR / F_EV, parse_dates=['event_time'])
raw = pd.read_csv(DATA_DIR / F_RAW)
reviews = pd.read_csv(DATA_DIR / F_REV)
selection = pd.read_csv(Path(F_SEL))
cutoffs = pd.read_csv(Path(F_CUT), index_col=0)
decisions = pd.read_csv(Path(F_DEC), index_col=0)
audit = pd.read_csv(Path(F_AUD), index_col=0)
reset_sens = pd.read_csv(Path(F_RES), index_col=[0, 1])
params = pd.read_csv(Path(F_PAR), index_col=0)['value']
TOP_SHARE = float(params['top_model_share_pct'])
TOP_RULE = f'at least {TOP_SHARE:g}% of its brand\'s considerations'

for d in (cons, top, rest):
    d['november'] = (d['month'] == '2019-11').astype(int)
    d['first_tv'] = (d['position_in_session'] == 'first_tv').astype(int)
    d['bought_other_brand'] = d['session_result'].isin(['bought_lg', 'bought_samsung', 'bought_other_four']).astype(int)
    d['price_band'] = pd.cut(d['price'], BANDS, labels=BAND_NAMES)
    d['main_rival_opened_before'] = np.select([d['brand'] == 'lg', d['brand'] == 'samsung'],
                                              [d['samsung_opened_before'], d['lg_opened_before']], 0)
BRANDS = cons['brand'].value_counts().index.tolist()          # largest first
ls = cons[cons['brand_group'].isin(['lg', 'samsung'])].copy()  # LG and Samsung considerations

# ---------------------------------------------------------------------------------------------
# Section 1. The six TV brands
# ---------------------------------------------------------------------------------------------
b = rates(cons, 'brand').loc[BRANDS]
b['share_cons'] = 100 * b['rows'] / b['rows'].sum()
b['share_purchases'] = 100 * b['events'] / b['events'].sum()
b['models'] = cons.groupby('brand')['product_id'].nunique()
b['cons_per_model'] = b['rows'] / b['models']
t11 = b[['rows', 'share_cons', 'events', 'share_purchases', 'rate_pct', 'ci_low_pct', 'ci_high_pct', 'models', 'cons_per_model']]
t11.columns = ['Considerations', 'Share of TV considerations (%)', 'Purchased considerations', 'Share of purchased TV considerations (%)',
               'Conversion (%)', 'Conversion, 95% CI low (%)', 'Conversion, 95% CI high (%)',
               'Models with at least one consideration', 'Considerations per model']
t11.index = [NAMES[x] for x in t11.index]
t11.index.name = 'Brand'
save_table('1.1', 'Traffic and conversion of the six TV brands at the retailer', t11,
           f'{UNIT} Conversion is the share of considerations bought in the same session. Intervals use standard errors clustered '
           'by user and by model.', [F_CONS], feeds='Report Table 3')

fig, ax = plt.subplots(figsize=(7.5, 3.6), layout='constrained')
y = np.arange(len(BRANDS))
ax.barh(y - 0.19, b['share_cons'], 0.38, color='#C9CDD3', label='Share of TV considerations')
ax.barh(y + 0.19, b['share_purchases'], 0.38, color='#2F3542', label='Share of purchased TV considerations')
for i, br in enumerate(BRANDS):
    ax.text(b.loc[br, 'share_cons'] + 0.5, i - 0.19, f"{b.loc[br, 'share_cons']:.1f}%", va='center', fontsize=7.5)
    ax.text(b.loc[br, 'share_purchases'] + 0.5, i + 0.19, f"{b.loc[br, 'share_purchases']:.1f}%", va='center', fontsize=7.5)
ax.set_yticks(y, [NAMES[x] for x in BRANDS])
for lab in ax.get_yticklabels():
    if lab.get_text() == 'LG':
        lab.set_color(COLORS['lg']); lab.set_fontweight('bold')
ax.invert_yaxis()
ax.xaxis.set_major_formatter(pct)
ax.set_xlabel('Share of all TV considerations, or of all purchased TV considerations, at the retailer')
ax.set_xlim(0, b[['share_cons', 'share_purchases']].max().max() * 1.15)
ax.legend(loc='lower right')
ax.grid(axis='y', visible=False)
finish(fig, '1.1', 'Share of TV considerations and of purchased TV considerations by brand',
       f'Each brand\'s share of TV considerations is shown next to its share of purchased TV considerations. {UNIT}',
       [F_CONS], feeds='Report Figure 1')

order = b.sort_values('rate_pct').index.tolist()
fig, ax = plt.subplots(figsize=(7.5, 3.2), layout='constrained')
for i, br in enumerate(order):
    r = b.loc[br]
    ax.errorbar(r['rate_pct'], i, xerr=[[r['rate_pct'] - r['ci_low_pct']], [r['ci_high_pct'] - r['rate_pct']]],
                fmt='o', color=COLORS[br], ms=6, capsize=3, elinewidth=1.2)
    ax.text(r['ci_high_pct'] + 0.05, i, f"{r['rate_pct']:.2f}%", va='center', fontsize=7.5)
overall = 100 * cons['purchased'].mean()
ax.axvline(overall, color='#999999', ls='--', lw=0.8)
ax.text(overall, len(order) - 0.45, f'All TV considerations {overall:.2f}%', fontsize=7.5, color='#666666', ha='center')
ax.set_yticks(range(len(order)), [NAMES[x] for x in order])
ax.xaxis.set_major_formatter(pct)
ax.set_xlabel('Conversion (share of considerations bought in the same session)')
ax.set_ylim(-0.6, len(order) - 0.2)
ax.grid(axis='y', visible=False)
finish(fig, '1.2', 'Conversion by TV brand', f'LG in red, Samsung in blue, the other four brands in grey. {CI_NOTE} {UNIT}',
       [F_CONS], feeds='Report Figure 1 (simpler option)')

# ---------------------------------------------------------------------------------------------
# Section 3. Data
# ---------------------------------------------------------------------------------------------
dups = int(raw.duplicated().sum())
t31 = pd.Series({
    'Raw events in the original file': len(raw),
    'Exact duplicate events removed': dups,
    'Events after cleaning': len(raw) - dups,
    'TV events, six brands (tv_events_clean.csv)': len(events),
    'TV considerations, six brands (tv_considerations.csv)': len(cons),
    'LG considerations': int((cons['brand_group'] == 'lg').sum()),
    'Samsung considerations': int((cons['brand_group'] == 'samsung').sum()),
    'Considerations of the other four brands': int((cons['brand_group'] == 'other_four').sum()),
    'Considerations of LG and Samsung top models (tv_considerations_top_models.csv)': len(top),
    'All other considerations (tv_considerations_without_top_models.csv)': len(rest),
    'Users with a TV consideration': cons['user_id'].nunique(),
    'Sessions with a TV consideration': cons[SESSION].drop_duplicates().shape[0],
    'Sessions with an LG or Samsung consideration': ls[SESSION].drop_duplicates().shape[0],
    'LG and Samsung considerations ending in a purchase': int(ls['purchased'].sum()),
}, name='Count').to_frame()
t31.index.name = 'Step'
save_table('3.1', 'Sample construction', t31, UNIT, [F_RAW, F_EV, F_CONS, F_TOP, F_REST], feeds='Report Appendix A1')

VARS = [  # column, label, statistic, definition
    ('purchased', 'Bought this model (outcome)', 'share', 'This model bought at or after its first view, in the same session'),
    ('added_to_cart', 'Added this model to the cart', 'share', 'Added at or after its first view, in the same session'),
    ('purchased_after_cart', 'Bought after adding to the cart', 'share', 'Bought after the first cart event for this model'),
    ('price', 'Price at the first view (median)', 'median', 'Retailer currency'),
    ('november', 'First view in November', 'share', 'Month of the first view'),
    ('top_model', 'Top model of its brand', 'share', f'Model holds {TOP_RULE} (LG and Samsung only)'),
    ('first_tv', 'First TV opened in the session', 'share', 'No other TV opened earlier in the session'),
    ('tvs_opened_before', 'TVs opened earlier in the session (mean)', 'mean', 'Other TV models opened earlier in the session'),
    ('other_brand_opened_before', 'Another brand opened earlier in the session', 'share', 'A TV of any other brand opened earlier in the session'),
    ('lg_opened_before', 'An LG TV opened earlier in the session', 'share', 'For LG considerations, another LG model'),
    ('samsung_opened_before', 'A Samsung TV opened earlier in the session', 'share', 'For Samsung considerations, another Samsung model'),
    ('other_four_opened_before', 'A TV of the other four brands opened earlier in the session', 'share', 'Artel, Haier, Kivi or TCL'),
    ('bought_other_brand', 'Bought another brand later in the session', 'share', 'A TV of another brand bought after this model was opened'),
    ('returning_visitor', 'Returning visitor', 'share', 'The user had an earlier session on the site, any category'),
    ('bought_tv_before', 'User bought a TV before', 'share', 'A TV purchase by the same user before this consideration, any session'),
]
rows_ = []
for col, lab, stat, definition in VARS:
    row = {'Variable': lab, 'Definition': definition,
           'Statistic': {'share': 'Share of considerations (%)', 'mean': 'Mean', 'median': 'Median'}[stat]}
    for g in GROUPS + ['all']:
        d = cons if g == 'all' else cons[cons['brand_group'] == g]
        row[NAMES.get(g, 'All TV considerations')] = (100 * d[col].mean() if stat == 'share'
                                                      else d[col].median() if stat == 'median' else d[col].mean())
    rows_.append(row)
t32 = pd.DataFrame(rows_).set_index('Variable')
save_table('3.2', 'Variables and their values by brand group', t32,
           f'Shares are percentages of considerations, means and medians are in the variable\'s own units. Earlier in the session means '
           f'before this model was opened, anywhere in the same session. {UNIT}', [F_CONS], feeds='Report Table 1')

# measurement checks
pe = events[events['event_type'] == 'purchase']
pk = pe[KEY + ['brand', 'event_time']].join(cons.set_index(KEY)['first_view'], on=KEY)
pk['captured'] = pk['first_view'] <= pk['event_time']
cl = cons.join(pe.groupby(['user_id', 'product_id'])['event_time'].max().rename('last_buy_any_session'), on=['user_id', 'product_id'])
cl['bought_later_only'] = ((cl['purchased'] == 0) & (cl['last_buy_any_session'] > cl['first_view'])).astype(int)


def measures(br):
    sel = (lambda d: d) if br == 'all' else (lambda d: d[d['brand'] == br])
    c, p_, l_, k_ = sel(cons), sel(pe), sel(cl), sel(pk)
    first = c.groupby('user_id')['first_view'].min()
    last = p_.groupby('user_id')['event_time'].max().reindex(first.index)
    rep = c[c['bought_tv_before'] == 1]
    return pd.Series({
        'TV view events': int((sel(events)['event_type'] == 'view').sum()),
        'Considerations': len(c),
        'Purchase events': len(p_),
        'Distinct user x session x model purchases': p_[KEY].drop_duplicates().shape[0],
        'Purchased considerations': int(c['purchased'].sum()),
        'Distinct buyers': p_['user_id'].nunique(),
        'Users who opened the brand': len(first),
        'Sessions that opened the brand': c[SESSION].drop_duplicates().shape[0],
        'Conversion per consideration (%)': 100 * c['purchased'].mean(),
        'Conversion per user (%)': 100 * (last >= first).mean(),
        'Purchase events preceded by a same-session view of the model (%)': 100 * k_['captured'].mean(),
        'Considerations bought only in a later session': int(l_['bought_later_only'].sum()),
        'Conversion counting purchases in later sessions (%)': 100 * (l_['purchased'] | l_['bought_later_only']).mean(),
        'Considerations by users who bought a TV before': len(rep),
        'Conversion of those considerations (%)': 100 * rep['purchased'].mean(),
        'Conversion of all other considerations (%)': 100 * c.loc[c['bought_tv_before'] == 0, 'purchased'].mean()})


t33 = pd.DataFrame({'LG': measures('lg'), 'Samsung': measures('samsung')})
t33['LG / Samsung'] = t33['LG'] / t33['Samsung']
t33['All six brands'] = measures('all')
t33.index.name = 'Measure'
save_table('3.3', 'Measurement checks on the unit, the outcome window and repeat buyers', t33,
           'Purchase events include repeated purchase events for the same model in the same session. Conversion per user is the share '
           'of users who opened the brand and bought a TV of that brand afterwards. A purchase in a later session is a purchase of the same '
           f'model by the same user after the first view, outside the session of the consideration. {UNIT}', [F_EV, F_CONS],
           feeds='Report Section 3 and Appendix A1')

dec = decisions.drop(columns='Decision').copy()
dec.insert(0, 'Decision', decisions['Decision'])
save_table('3.4', 'Conversion by each variable kept in the dataset and by each variable left out', dec,
           'Computed by 01_build_tv_dataset.py on LG and Samsung considerations. Each row compares the considerations in the group with '
           'the reference group named in the row. The rows on earlier sessions use considerations from '
           f"{pd.Timestamp(params['history_check_start']):%d %B}" ' onward, when seven days of the data have elapsed, for considerations with no '
           'earlier TV purchase and with no other brand opened earlier in the same session. The course file is a curated sample, so a '
           'user\'s earlier sessions can be missing. A difference in this table does not by itself decide whether a variable belongs in '
           'a model.',
           [F_DEC], feeds='Report Section 3 (choice of variables)')

periods = pd.cut(ls['first_view'], [pd.Timestamp('2019-10-01', tz='UTC'), pd.Timestamp('2019-11-01', tz='UTC'),
                                    pd.Timestamp('2019-11-25', tz='UTC'), pd.Timestamp('2019-12-01', tz='UTC')],
                 labels=['October', '1 to 24 November', '25 to 30 November'], right=False)
lp = ls.assign(period=periods)
t35 = pd.DataFrame({
    'LG considerations': lp[lp['brand'] == 'lg'].groupby('period', observed=True).size(),
    'Samsung considerations': lp[lp['brand'] == 'samsung'].groupby('period', observed=True).size(),
    'LG sessions': lp[lp['brand'] == 'lg'].drop_duplicates(SESSION).groupby('period', observed=True).size(),
    'Samsung sessions': lp[lp['brand'] == 'samsung'].drop_duplicates(SESSION).groupby('period', observed=True).size(),
    'LG conversion (%)': 100 * lp[lp['brand'] == 'lg'].groupby('period', observed=True)['purchased'].mean(),
    'Samsung conversion (%)': 100 * lp[lp['brand'] == 'samsung'].groupby('period', observed=True)['purchased'].mean()})
t35['LG sessions per Samsung session'] = t35['LG sessions'] / t35['Samsung sessions']
t35['LG minus Samsung conversion (pp)'] = t35['LG conversion (%)'] - t35['Samsung conversion (%)']
t35.index.name = 'Period of the first view'
save_table('3.5', 'LG and Samsung traffic and conversion by period', t35,
           f'The last period is the final week of the data. {UNIT}', [F_CONS], feeds='Report Section 3')

wk = cons.assign(week=cons['first_view'].dt.tz_convert(None).dt.to_period('W-SUN').dt.start_time)
wr = rates(wk, ['week', 'brand_group']).reset_index()
fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.3), layout='constrained')
for g in GROUPS:
    d = wr[wr['brand_group'] == g]
    axes[0].plot(d['week'], d['rows'], marker='o', ms=3, color=COLORS[g], label=NAMES[g])
    axes[1].plot(d['week'], d['rate_pct'], marker='o', ms=3, color=COLORS[g], label=NAMES[g])
for ax in axes:
    ax.axvline(pd.Timestamp('2019-11-01'), color='#999999', ls='--', lw=0.8)
    ax.tick_params(axis='x', rotation=45, labelsize=7.5)
    ax.set_xlabel('Week starting (Monday)')
axes[0].set_title('A. Considerations per week')
axes[0].set_ylabel('Considerations')
axes[1].set_title('B. Conversion per week')
axes[1].set_ylabel('Conversion')
axes[1].yaxis.set_major_formatter(pct)
axes[0].legend()
d0, d1 = cons['first_view'].min(), cons['first_view'].max()
finish(fig, '3.1', 'Weekly TV traffic and conversion by brand group',
       f"The dashed line marks 1 November. The first week runs from {d0:%d} to {wk['week'].min() + pd.Timedelta(days=6):%d %B} and the "
       f"last from {wk['week'].max():%d} to {d1:%d %B}, so both are partial. {UNIT}", [F_CONS],
       feeds='Report Section 3 (support for the month control)')

aud = audit.copy()
aud['Value'] = aud['Value'].map(lambda x: f'{float(x):,.0f}' if float(x).is_integer() else f'{float(x):,.2f}')
aud['Note'] = aud['Note'].fillna('')
save_table('3.6', 'Data audit', aud,
           'Computed by 01_build_tv_dataset.py before the datasets are saved. Counts of sessions and models opened describe the data and '
           'cannot by themselves show whether any session was automated.', [F_AUD], feeds='Report Section 3 and Appendix A1')

rst = reset_sens.copy()
rst.index = rst.index.set_levels([rst.index.levels[0], [POSITION[p] for p in rst.index.levels[1]]])
rst.columns = [c.replace('lg ', 'LG ').replace('samsung ', 'Samsung ').replace('other_four ', 'Other four brands, ') for c in rst.columns]
save_table('3.7', 'Position in the session over the whole session and since the last TV purchase in the session', rst,
           'The dataset uses the whole session. The second definition restarts the history after a TV purchase earlier in the same '
           f"session, which changes the position of {int(float(audit.loc['Considerations opened after a TV purchase earlier in the same session', 'Value'])):,} "
           'considerations at most. Computed by 01_build_tv_dataset.py.', [F_RES], feeds='Report Section 3 (sensitivity)')

# ---------------------------------------------------------------------------------------------
# Section 4.1. Why TVs: the six product categories, built from the original file with the same consideration rules
# ---------------------------------------------------------------------------------------------
ev = raw.drop_duplicates().copy()
ev['event_time'] = pd.to_datetime(ev['event_time'], utc=True)
v = ev[ev['event_type'] == 'view']
allrows = (v.sort_values('event_time', kind='stable').drop_duplicates(KEY)
           [KEY + ['brand', 'product_category', 'event_time', 'price']].rename(columns={'event_time': 'first_view'}))
lastp = ev[ev['event_type'] == 'purchase'].groupby(KEY)['event_time'].max().rename('last_purchase')
allrows = allrows.join(lastp, on=KEY)
allrows['purchased'] = (allrows['last_purchase'] >= allrows['first_view']).astype(int)
assert len(allrows[allrows['product_category'] == 'tv']) == len(cons)

cat = []
for c, d in allrows.groupby('product_category'):
    lg, sm = d[d['brand'] == 'lg'], d[d['brand'] == 'samsung']
    gap, se = gap_precision(d)
    f = d[d['brand'].isin(['lg', 'samsung'])].groupby(SESSION + ['brand'])['first_view'].min().unstack().dropna()
    cat.append({'Category': CATEGORIES[c], 'All-brand considerations': len(d), 'All-brand median price': d['price'].median(),
                'LG considerations': len(lg), 'LG purchased considerations': int(lg['purchased'].sum()),
                'LG models': lg['product_id'].nunique(), 'Samsung models': sm['product_id'].nunique(),
                'LG conversion (%)': 100 * lg['purchased'].mean(), 'Samsung conversion (%)': 100 * sm['purchased'].mean(),
                'LG median price': lg['price'].median(), 'Samsung median price': sm['price'].median(),
                'LG minus Samsung (pp)': gap,
                'Gap, 95% CI low (pp)': gap - 1.96 * se['user_product'], 'Gap, 95% CI high (pp)': gap + 1.96 * se['user_product'],
                'z, clustered by user and model': gap / se['user_product'],
                'z, clustered by model only (sensitivity)': gap / se['product'],
                'z, considerations independent (sensitivity)': gap / se['independent'],
                'Samsung opened first in two-brand sessions (%)': 100 * (f['samsung'] < f['lg']).mean()})
t41 = pd.DataFrame(cat).set_index('Category').sort_values('LG considerations', ascending=False)
save_table('4.1', 'LG against Samsung in the six product categories', t41,
           'Same consideration rules as the TV file, applied to every category. Intervals and the main z statistic use standard errors '
           'clustered by user and by model. These are exploratory comparisons made after looking at all six categories. A gap that is '
           'significant in one category and not in another does not show that the two gaps differ. The last column is the share of '
           'sessions opening both brands in which a Samsung model was opened first.', [F_RAW], feeds='Report Table 2 and Figure 5')

fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.2), layout='constrained', sharey=True)
yy = np.arange(len(t41))
colors = [COLORS['lg'] if i == 'TV' else '#B9BCC1' for i in t41.index]
axes[0].barh(yy, t41['LG considerations'], color=colors)
for i, val in enumerate(t41['LG considerations']):
    axes[0].text(val + 300, i, f'{val:,}', va='center', fontsize=7.5)
axes[0].set_title('A. LG considerations')
axes[0].set_xlabel('Considerations')
axes[0].set_xlim(0, t41['LG considerations'].max() * 1.25)
axes[0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
lo_, hi_ = t41['Gap, 95% CI low (pp)'], t41['Gap, 95% CI high (pp)']
axes[1].errorbar(t41['LG minus Samsung (pp)'], yy, xerr=[t41['LG minus Samsung (pp)'] - lo_, hi_ - t41['LG minus Samsung (pp)']],
                 fmt='none', ecolor='#555555', elinewidth=1, capsize=3)
axes[1].scatter(t41['LG minus Samsung (pp)'], yy, c=colors, s=36, zorder=3)
axes[1].axvline(0, color='#333333', lw=0.8)
axes[1].set_title('B. LG minus Samsung conversion')
axes[1].set_xlabel('Percentage points')
axes[0].set_yticks(yy, t41.index)
axes[0].invert_yaxis()
for ax in axes:
    ax.grid(axis='y', visible=False)
finish(fig, '4.1', 'LG sample size and conversion gap with Samsung by category',
       'TVs in red. Categories ordered by LG considerations. Intervals are 95% intervals with standard errors clustered by user and '
       'by model. Exploratory comparisons made after looking at all six categories.',
       [F_RAW], feeds='Report Section 4.1')

# ---------------------------------------------------------------------------------------------
# Section 4.2. Why Samsung: price positioning
# ---------------------------------------------------------------------------------------------
lo, hi = cons.loc[cons['brand'] == 'lg', 'price'].quantile([.05, .95])
q = cons.groupby('brand')['price'].quantile([.05, .25, .5, .75, .95]).unstack().loc[BRANDS]
q.columns = ['5th percentile', '25th percentile', 'Median', '75th percentile', '95th percentile']
q['Considerations inside the LG 5th to 95th percentile range (%)'] = 100 * cons.groupby('brand')['price'].apply(lambda s: s.between(lo, hi).mean())
q.index = [NAMES[x] for x in q.index]
q.index.name = 'Brand'
save_table('4.2', 'Price at the first view of TV considerations by brand', q,
           f'Prices are in the retailer\'s unidentified currency. The LG range runs from {lo:,.0f} to {hi:,.0f}. {UNIT}', [F_CONS],
           feeds='Report Table 3')

fig, ax = plt.subplots(figsize=(7.5, 3.4), layout='constrained')
bp = ax.boxplot([cons.loc[cons['brand'] == br, 'price'] for br in BRANDS], whis=(5, 95), showfliers=False, patch_artist=True,
                widths=0.55, medianprops={'color': 'white', 'lw': 1.4})
for patch, br in zip(bp['boxes'], BRANDS):
    patch.set_facecolor(COLORS[br]); patch.set_edgecolor(COLORS[br])
ax.axhspan(lo, hi, color=COLORS['lg'], alpha=0.07, zorder=0)
ax.text(len(BRANDS) + 0.45, np.sqrt(lo * hi), 'LG 5th to 95th\npercentile range', fontsize=7.5, color=COLORS['lg'], va='center')
ax.set_yscale('log')
ax.set_yticks([100, 150, 200, 300, 500, 700, 1000, 1500, 2000])
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xticks(range(1, len(BRANDS) + 1), [NAMES[x] for x in BRANDS])
ax.set_xlim(0.4, len(BRANDS) + 1.6)
ax.set_ylabel('Price at the first view (log scale, retailer currency)')
ax.grid(axis='x', visible=False)
finish(fig, '4.2', 'Price distribution of TV considerations by brand',
       f'Boxes span the 25th to 75th percentiles, the white line is the median and whiskers reach the 5th and 95th percentiles. {UNIT}',
       [F_CONS], feeds='Report Section 4.2')

# ---------------------------------------------------------------------------------------------
# Section 4.3. Funnel stages
# ---------------------------------------------------------------------------------------------
two_brand = ls.groupby(SESSION)['brand'].nunique().pipe(lambda s: s[s == 2].index)
ls['two_brand_session'] = ls.set_index(SESSION).index.isin(two_brand).astype(int)


def funnel(d):
    n, carts, pac, buys = len(d), d['added_to_cart'].sum(), d['purchased_after_cart'].sum(), d['purchased'].sum()
    return pd.Series({'Considerations': n, 'View to cart (%)': 100 * carts / n, 'Cart to purchase (%)': 100 * pac / carts,
                      'Conversion (%)': 100 * buys / n, 'Purchases without a qualifying cart-to-purchase sequence': int(buys - pac)})


f_rows = {('All sessions', NAMES[br]): funnel(cons[cons['brand'] == br]) for br in BRANDS}
for br in ['lg', 'samsung']:
    f_rows[('Sessions that opened both LG and Samsung', NAMES[br])] = funnel(ls[(ls['brand'] == br) & (ls['two_brand_session'] == 1)])
t43 = pd.DataFrame(f_rows).T
t43.index.names = ['Sample', 'Brand']
save_table('4.3', 'Funnel stages by brand, for all sessions and for sessions that opened both LG and Samsung', t43,
           f'Cart and purchase must happen at or after the first view, in the same session. {UNIT}', [F_CONS], feeds='Report Appendix A3')

fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.2), layout='constrained', sharey=True)
stages = ['View to cart (%)', 'Cart to purchase (%)']
for ax, sample, title in [(axes[0], 'All sessions', 'A. All sessions'),
                          (axes[1], 'Sessions that opened both LG and Samsung', 'B. Sessions that opened both brands')]:
    x = np.arange(len(stages))
    for j, br in enumerate(['LG', 'Samsung']):
        vals = t43.loc[(sample, br), stages].astype(float)
        ax.bar(x + (j - 0.5) * 0.36, vals, 0.36, color=COLORS[br.lower()], label=br)
        for xi, val in zip(x + (j - 0.5) * 0.36, vals):
            ax.text(xi, val + 0.8, f'{val:.1f}%', ha='center', fontsize=7.5)
    ax.set_xticks(x, ['View to cart', 'Cart to purchase'])
    ax.set_ylim(0, t43[stages].astype(float).max().max() * 1.15)
    ax.set_title(title)
    ax.yaxis.set_major_formatter(pct)
    ax.grid(axis='x', visible=False)
axes[0].set_ylabel('Stage rate')
axes[0].legend(loc='upper left')
finish(fig, '4.3', 'Funnel stage rates for LG and Samsung', UNIT, [F_CONS], feeds='Report Section 4.3')

# ---------------------------------------------------------------------------------------------
# Section 4.4. Shopper mix: how much of the conversion gap comes from the kind of shopper who reaches each brand
# ---------------------------------------------------------------------------------------------
t44a = {}
for val, name in [(0, 'No other brand opened earlier in the session'), (1, 'Another brand opened earlier in the session')]:
    d = ls[ls['other_brand_opened_before'] == val]
    t44a[name] = {'Share of LG considerations (%)': 100 * (d['brand'] == 'lg').sum() / (ls['brand'] == 'lg').sum(),
                  'Share of Samsung considerations (%)': 100 * (d['brand'] == 'samsung').sum() / (ls['brand'] == 'samsung').sum(),
                  'LG conversion (%)': 100 * d.loc[d['brand'] == 'lg', 'purchased'].mean(),
                  'Samsung conversion (%)': 100 * d.loc[d['brand'] == 'samsung', 'purchased'].mean()}
t44a['All considerations'] = {'Share of LG considerations (%)': 100.0, 'Share of Samsung considerations (%)': 100.0,
                              'LG conversion (%)': 100 * ls.loc[ls['brand'] == 'lg', 'purchased'].mean(),
                              'Samsung conversion (%)': 100 * ls.loc[ls['brand'] == 'samsung', 'purchased'].mean()}
t44a = pd.DataFrame(t44a).T
t44a['LG minus Samsung (pp)'] = t44a['LG conversion (%)'] - t44a['Samsung conversion (%)']
t44a.index.name = 'Shopper type'
save_table('4.4a', 'Conversion of LG and Samsung by whether the shopper had opened another brand earlier in the session', t44a,
           f'Shoppers are grouped by whether a TV of another brand was opened earlier in the same session. {UNIT}', [F_CONS],
           feeds='Report Section 4.4')


def mix_split(d, cols):
    """LG minus Samsung conversion split into shopper mix and the gap within the same type of shopper."""
    lg, sm = d[d['brand'] == 'lg'], d[d['brand'] == 'samsung']
    rl, rs = lg.groupby(cols, observed=True)['purchased'].mean(), sm.groupby(cols, observed=True)['purchased'].mean()
    wl, ws = lg.groupby(cols, observed=True).size() / len(lg), sm.groupby(cols, observed=True).size() / len(sm)
    common = rl.index.intersection(rs.index)
    lg_at_sm_mix = (ws[common] * rl[common]).sum() / ws[common].sum()
    sm_at_lg_mix = (wl[common] * rs[common]).sum() / wl[common].sum()
    LG, SM = lg['purchased'].mean(), sm['purchased'].mean()
    return {'Raw gap (pp)': 100 * (LG - SM), 'Composition contribution (pp)': 100 * (LG - lg_at_sm_mix),
            'Part within the same type of shopper (pp)': 100 * (lg_at_sm_mix - SM),
            'Composition share of the raw gap (%)': 100 * (LG - lg_at_sm_mix) / (LG - SM),
            'Composition share, Samsung reweighted to LG\'s mix (%)': 100 * (sm_at_lg_mix - SM) / (LG - SM),
            'Groups compared': len(common), 'Samsung considerations in groups that also hold LG considerations (%)': 100 * ws[common].sum()}


SPLITS = [('Another brand opened earlier in the session (yes or no)', ['other_brand_opened_before']),
          ('Position in the session (three groups)', ['position_in_session']),
          ('Position, price band, month and returning visitor together', ['position_in_session', 'price_band', 'month', 'returning_visitor'])]
mix = []
for name, cols in SPLITS:
    r = {'Shopper type defined by': name, **mix_split(ls, cols)}
    r['Composition share, excluding considerations after an earlier recorded TV purchase (%)'] = mix_split(ls[ls['bought_tv_before'] == 0], cols)['Composition share of the raw gap (%)']
    mix.append(r)
t44b = pd.DataFrame(mix).set_index('Shopper type defined by')
save_table('4.4b', 'The LG minus Samsung conversion gap split into shopper mix and the gap within the same type of shopper', t44b,
           'The shopper-mix part is LG\'s conversion minus the conversion LG would have if its considerations were spread across shopper '
           'types as Samsung\'s are. The other part compares the two brands within the observed groups; individual shoppers are not held fixed. The '
           'second share does the reweighting the other way round. The last row uses every combination of the four characteristics as a '
           f'shopper type. This is a descriptive reweighting, not a causal attribution. {UNIT}', [F_CONS], feeds='Report Appendix A4')

# ---------------------------------------------------------------------------------------------
# Section 4.5. The shopper's state when the model is opened
# ---------------------------------------------------------------------------------------------
JUST = [('other_brand_opened_before', 'Another brand opened earlier in the session'),
        ('main_rival_opened_before', 'The other main brand opened earlier in the session (Samsung for LG, LG for Samsung)'),
        ('other_four_opened_before', 'A TV of the other four brands opened earlier in the session'),
        ('first_tv', 'First TV opened in the session'),
        ('returning_visitor', 'Returning visitor'),
        ('bought_tv_before', 'User bought a TV before'),
        ('november', 'First view in November')]
jr = []
for col, lab in JUST:
    for g in ['lg', 'samsung']:
        d = ls[ls['brand'] == g]
        c0, c1 = d.loc[d[col] == 0, 'purchased'], d.loc[d[col] == 1, 'purchased']
        jr.append({'Variable': lab, 'Brand': NAMES[g], 'Share of considerations with yes (%)': 100 * d[col].mean(),
                   'Conversion if no (%)': 100 * c0.mean(), 'Conversion if yes (%)': 100 * c1.mean(),
                   'Difference, yes minus no (pp)': 100 * (c1.mean() - c0.mean()), 'Considerations with yes': len(c1)})
t45a = pd.DataFrame(jr).set_index(['Variable', 'Brand'])
save_table('4.5a', 'Conversion by each yes-or-no variable, LG and Samsung', t45a,
           'A large difference between yes and no, together with a different share of yes for LG and Samsung, is what makes a variable '
           f'matter for the comparison of the two brands. {UNIT}', [F_CONS], feeds='Report Section 4.5 and Table 1')

pos_order = list(POSITION)
pr = rates(cons, ['position_in_session', 'brand_group'])
share = pd.crosstab(cons['position_in_session'], cons['brand_group'], normalize='columns') * 100
t45b = pd.DataFrame({f'{NAMES[g]} conversion (%)': pr.xs(g, level='brand_group')['rate_pct'] for g in GROUPS}).loc[pos_order]
t45b['LG minus Samsung (pp)'] = t45b['LG conversion (%)'] - t45b['Samsung conversion (%)']
for g in GROUPS:
    t45b[f'Share of {NAMES[g]} considerations (%)'] = share.loc[pos_order, g]
t45b.index = [POSITION[p] for p in pos_order]
t45b.index.name = 'Position in the session'
save_table('4.5b', 'Conversion by position in the session', t45b, UNIT, [F_CONS], feeds='Report Table 4')

fig, ax = plt.subplots(figsize=(7.5, 3.4), layout='constrained')
x = np.arange(len(pos_order))
for j, g in enumerate(GROUPS):
    errbar_bars(ax, x + (j - 1) * 0.26, pr.xs(g, level='brand_group').loc[pos_order], COLORS[g], 0.26, NAMES[g])
ax.set_xticks(x, ['First TV opened\nin the session', 'After other models of\nthe same brand only', 'After a TV of\nanother brand'])
ax.set_ylim(0, pr['ci_high_pct'].max() * 1.2)
ax.yaxis.set_major_formatter(pct)
ax.set_ylabel('Conversion')
ax.legend()
ax.grid(axis='x', visible=False)
finish(fig, '4.5a', 'Conversion by position in the session and brand group',
       f'Only TVs opened earlier in the same session count. {CI_NOTE} {UNIT}',
       [F_CONS], feeds='Report Table 4')

ls['opened_before'] = np.select(
    [ls['tvs_opened_before'] == 0,
     ls['other_brand_opened_before'] == 0,
     (ls['main_rival_opened_before'] == 1) & (ls['other_four_opened_before'] == 0),
     (ls['main_rival_opened_before'] == 0) & (ls['other_four_opened_before'] == 1)],
    ['No TV', 'Only other models of the same brand', 'The other main brand, none of the other four',
     'One of the other four brands, not the other main brand'],
    'Both the other main brand and one of the other four')
ob_order = ['No TV', 'Only other models of the same brand', 'The other main brand, none of the other four',
            'One of the other four brands, not the other main brand', 'Both the other main brand and one of the other four']
t45c = {}
for g in ['lg', 'samsung']:
    d = ls[ls['brand'] == g]
    t45c[f'Share of {NAMES[g]} considerations (%)'] = 100 * d['opened_before'].value_counts(normalize=True)
    t45c[f'{NAMES[g]} conversion (%)'] = 100 * d.groupby('opened_before')['purchased'].mean()
    t45c[f'{NAMES[g]} considerations'] = d['opened_before'].value_counts()
t45c = pd.DataFrame(t45c).loc[ob_order]
t45c['LG minus Samsung (pp)'] = t45c['LG conversion (%)'] - t45c['Samsung conversion (%)']
t45c.index.name = 'TVs opened earlier in the session'
save_table('4.5c', 'Conversion by which brands the shopper had opened earlier in the session', t45c,
           f'The other main brand is Samsung for LG considerations and LG for Samsung considerations. {UNIT}', [F_CONS],
           feeds='Report Section 4.5')

res_order = ['bought_this_model', 'bought_lg', 'bought_samsung', 'bought_other_four', 'bought_other_model_same_brand', 'bought_nothing']
# head-to-head sessions, counted once per session: which brand was opened first and which brands were bought
pe_g = pe.assign(g=np.where(pe['brand'].isin(['lg', 'samsung']), pe['brand'], 'other_four'))
lg_t = ls[ls['brand'] == 'lg'].groupby(SESSION)['first_view'].min()
sm_t = ls[ls['brand'] == 'samsung'].groupby(SESSION)['first_view'].min()
h2h = pd.concat([lg_t.rename('lg'), sm_t.rename('samsung')], axis=1).dropna()
h2h['order'] = np.where(h2h['samsung'] < h2h['lg'], 'Samsung opened first', 'LG opened first')
pb = pe_g.groupby(SESSION + ['g']).size().unstack().reindex(h2h.index).reindex(columns=GROUPS).fillna(0) > 0
h2h['lg_not_sm'], h2h['sm_not_lg'] = (pb['lg'] & ~pb['samsung']).astype(int), (pb['samsung'] & ~pb['lg']).astype(int)
h2h['both'], h2h['neither'] = (pb['lg'] & pb['samsung']).astype(int), (~pb['lg'] & ~pb['samsung']).astype(int)
h2h['o4'] = pb['other_four'].astype(int)
t45d = {}
for order_, d in h2h.groupby('order'):
    u = d.index.get_level_values('user_id')
    r = {'Sessions': len(d), 'Users': u.nunique()}
    for col, lab in [('lg_not_sm', 'Bought LG, not Samsung'), ('sm_not_lg', 'Bought Samsung, not LG')]:
        m, lo_, hi_ = mean_ci(d[col], u)
        r[f'{lab} (%)'], r[f'{lab}, 95% CI low (%)'], r[f'{lab}, 95% CI high (%)'] = m, lo_, hi_
        r[f'{lab} (sessions)'] = int(d[col].sum())
    r['Bought both LG and Samsung (%)'] = 100 * d['both'].mean()
    r['Bought neither LG nor Samsung (%)'] = 100 * d['neither'].mean()
    r['Bought a TV of the other four brands (%)'] = 100 * d['o4'].mean()
    t45d[order_] = r
t45d = pd.DataFrame(t45d).T
t45d.index.name = 'Sessions that opened both LG and Samsung'
save_table('4.5d', 'Which brands were bought in sessions that opened both LG and Samsung, by which brand was opened first', t45d,
           'Each session is counted once. A purchase anywhere in the session counts, including one made before the second brand was '
           'opened. Intervals use standard errors clustered by user, since a user can have several sessions.',
           [F_CONS, F_EV], feeds='Report Section 4.5')

# ---------------------------------------------------------------------------------------------
# Section 4.6. After the first view: where shoppers go, and which brand they open first
# ---------------------------------------------------------------------------------------------
fresh = cons[cons['other_brand_opened_before'] == 0]
t46a = (pd.crosstab(fresh['session_result'], fresh['brand_group'], normalize='columns') * 100).reindex(res_order).fillna(0)[GROUPS]
t46a.columns = [f'{NAMES[g]} (%)' for g in GROUPS]
t46a.loc['considerations'] = fresh['brand_group'].value_counts()[GROUPS].values
t46a.index = [RESULT.get(i, 'Considerations') for i in t46a.index]
t46a.index.name = 'What the shopper bought after opening this model, same session'
save_table('4.6a', 'What was bought after a TV model was opened, for shoppers who had not opened another brand earlier', t46a,
           'A classification of considerations, not of sessions. Each consideration takes the first label that applies in the order '
           'listed (this model, then the first other brand bought, then another model of the same brand), so a session with several '
           f'considerations can count one purchase more than once. {UNIT}', [F_CONS], feeds='Report Section 4.6')

sess = (cons.sort_values('first_view', kind='stable').groupby(SESSION)
        .agg(first_group=('brand_group', 'first'), first_price=('price', 'first'),
             n_lg=('brand_group', lambda s: (s == 'lg').sum()), n_sm=('brand_group', lambda s: (s == 'samsung').sum()),
             n_o4=('brand_group', lambda s: (s == 'other_four').sum())))
lg_first_time = ls[ls['brand'] == 'lg'].groupby(SESSION)['first_view'].min()
sm_first_time = ls[ls['brand'] == 'samsung'].groupby(SESSION)['first_view'].min()
both = pd.concat([lg_first_time.rename('lg'), sm_first_time.rename('samsung')], axis=1).dropna()
both = both.join(sess[['n_lg', 'n_sm']])
both['samsung_first'] = (both['samsung'] < both['lg']).astype(int)
expected = both['n_sm'] / (both['n_sm'] + both['n_lg'])
sims = (np.random.default_rng(2026).random((2000, len(both))) < expected.to_numpy()).mean(axis=1)
one_each = both[(both['n_lg'] == 1) & (both['n_sm'] == 1)]
_, lo1, hi1 = mean_ci(both['samsung_first'], both.index.get_level_values('user_id'))
_, lo2, hi2 = mean_ci(one_each['samsung_first'], one_each.index.get_level_values('user_id'))
n_lg_s, n_sm_s = int((sess['n_lg'] > 0).sum()), int((sess['n_sm'] > 0).sum())
t46b = pd.Series({
    'Sessions that opened both LG and Samsung': len(both),
    'Samsung opened first (%)': 100 * both['samsung_first'].mean(),
    'Samsung opened first, 95% CI low (%)': lo1, 'Samsung opened first, 95% CI high (%)': hi1,
    'Expected if each shopper\'s models came in random order (%)': 100 * expected.mean(),
    'Random-order range, 2.5th percentile of 2,000 draws (%)': 100 * np.percentile(sims, 2.5),
    'Random-order range, 97.5th percentile of 2,000 draws (%)': 100 * np.percentile(sims, 97.5),
    'Sessions with exactly one model of each brand': len(one_each),
    'Samsung opened first among them (%)': 100 * one_each['samsung_first'].mean(),
    'Samsung opened first among them, 95% CI low (%)': lo2, 'Samsung opened first among them, 95% CI high (%)': hi2,
    'Expected among them if the order were random (%)': 50.0,
    'Samsung share of sessions that opened either brand (%)': 100 * n_sm_s / (n_sm_s + n_lg_s)}, name='Value').to_frame()
t46b.index.name = 'Measure'
save_table('4.6b', 'Which brand shoppers open first when they open both LG and Samsung in a session', t46b,
           'The population is the sessions in which both brands were opened, which is a selected group of TV searches. The random-order '
           'benchmark keeps the LG and Samsung models each shopper eventually opened and assumes their order could have been shuffled. '
           'It is a descriptive benchmark that removes the advantage Samsung gets from shoppers opening more Samsung models, and it says '
           'nothing about placement. Intervals use standard errors clustered by user.', [F_CONS], feeds='Report Section 4.6')

flows, rate_cols = [], {'lg': 'n_lg', 'samsung': 'n_sm', 'other_four': 'n_o4'}
for src_g in GROUPS:
    d = sess[sess['first_group'] == src_g]
    r = {'First TV of the session': NAMES[src_g], 'Sessions': len(d)}
    for dst in GROUPS:
        if dst != src_g:
            went = (d[rate_cols[dst]] > 0).astype(int)
            m_r, lo_r, hi_r = mean_ci(went, d.index.get_level_values('user_id'))
            r[f'Went on to {NAMES[dst]} (%)'] = m_r
            r[f'Went on to {NAMES[dst]}, 95% CI low (%)'], r[f'Went on to {NAMES[dst]}, 95% CI high (%)'] = lo_r, hi_r
    others = [rate_cols[g] for g in GROUPS if g != src_g]
    r['Went on to any other brand group (%)'] = 100 * (d[others] > 0).any(axis=1).mean()
    flows.append(r)
t46c = pd.DataFrame(flows).set_index('First TV of the session')
t46c = t46c[['Sessions'] + [c for g in GROUPS for c in t46c.columns if c.startswith(f'Went on to {NAMES[g]}')]
            + ['Went on to any other brand group (%)']]
save_table('4.6c', 'Where shoppers went after the first TV they opened in a session', t46c,
           'Each session is classified by the brand group of the first TV opened. Every rate uses the sessions that started with that '
           'brand group as its denominator. The other four brands count as one group. Intervals use standard errors clustered by user. '
           'These are recorded paths and say nothing about what caused them.', [F_CONS], feeds='Report Section 4.6')

sess['first_band'] = pd.cut(sess['first_price'], BANDS, labels=BAND_NAMES)
fb = []
for band in BAND_NAMES:
    for src_g, rival in [('lg', 'n_sm'), ('samsung', 'n_lg')]:
        d = sess[(sess['first_group'] == src_g) & (sess['first_band'] == band)]
        fb.append({'Price of the first TV': band, 'First TV of the session': NAMES[src_g], 'Sessions': len(d),
                   'Went on to the other main brand (%)': 100 * (d[rival] > 0).mean(),
                   'Went on to one of the other four brands (%)': 100 * (d['n_o4'] > 0).mean(),
                   'Went on to any other brand (%)': 100 * ((d[rival] > 0) | (d['n_o4'] > 0)).mean()})
t46d = pd.DataFrame(fb).set_index(['Price of the first TV', 'First TV of the session'])
save_table('4.6d', 'Where shoppers went after an LG or Samsung first TV, by the price of that TV', t46d,
           'The other main brand is Samsung after an LG first TV and LG after a Samsung first TV. Comparing the rates of going on to a '
           'TV of the other four brands holds the destination fixed. These are recorded paths and say nothing about what caused them.',
           [F_CONS], feeds='Report Section 4.6')

fig, ax = plt.subplots(figsize=(7.5, 3.3), layout='constrained')
for i, src_g in enumerate(GROUPS):
    dests = [g for g in GROUPS if g != src_g]
    for j, dst in enumerate(dests):
        val = t46c.loc[NAMES[src_g], f'Went on to {NAMES[dst]} (%)']
        lo_r = t46c.loc[NAMES[src_g], f'Went on to {NAMES[dst]}, 95% CI low (%)']
        hi_r = t46c.loc[NAMES[src_g], f'Went on to {NAMES[dst]}, 95% CI high (%)']
        xx = i + (j - 0.5) * 0.36
        ax.bar(xx, val, 0.36, color=COLORS[dst])
        ax.errorbar(xx, val, yerr=[[val - lo_r], [hi_r - val]], fmt='none', ecolor='#333333', elinewidth=0.8, capsize=2.5)
        ax.text(xx, hi_r + 0.4, f'{val:.1f}%', ha='center', fontsize=7.5)
ax.set_xticks(range(3), [f'First TV {NAMES[g]}\n({int(t46c.loc[NAMES[g], "Sessions"]):,} sessions)' for g in GROUPS])
ax.set_ylabel('Sessions that went on to the brand group')
ax.yaxis.set_major_formatter(pct)
ax.set_ylim(0, t46c.filter(like='CI high').max().max() * 1.35)
ax.legend([plt.Rectangle((0, 0), 1, 1, color=COLORS[g]) for g in GROUPS], [f'Went on to {NAMES[g]}' for g in GROUPS],
          loc='upper center', ncol=3)
ax.grid(axis='x', visible=False)
finish(fig, '4.6', 'Where shoppers went after the first TV they opened in a session',
       f'Data in Table 4.6c. {CI_NOTE_SESSIONS}', [F_CONS], feeds='Report Section 4.6')

# ---------------------------------------------------------------------------------------------
# Section 4.7. From conversion to reach, and the role of the top models
# ---------------------------------------------------------------------------------------------
FACTORS = ['sessions', 'per_session', 'conversion']


def factors(d):
    k = d.groupby('brand').agg(purchases=('purchased', 'sum'), considerations=('purchased', 'size'))
    k['sessions'] = d[SESSION + ['brand']].drop_duplicates().groupby('brand').size()
    k['per_session'] = k['considerations'] / k['sessions']
    k['conversion'] = 100 * k['purchases'] / k['considerations']
    return k


def shares(k):
    """Four ways to split LG's shortfall in purchased considerations among the three factors, in % of the shortfall."""
    L, S = k.loc['lg', FACTORS].to_numpy(float), k.loc['samsung', FACTORS].to_numpy(float)
    gap, idx = S.prod() - L.prod(), np.arange(3)
    log = np.log(S / L) / np.log(S.prod() / L.prod())
    shapley = np.zeros(3)
    for order_ in itertools.permutations(range(3)):
        cur = L.copy()
        for i in order_:
            before = cur.prod()
            cur[i] = S[i]
            shapley[i] += cur.prod() - before
    lg_gets = np.array([np.where(idx == i, S, L).prod() - L.prod() for i in range(3)])
    sm_loses = np.array([S.prod() - np.where(idx == i, L, S).prod() for i in range(3)])
    return 100 * log, 100 * shapley / 6 / gap, 100 * lg_gets / gap, 100 * sm_loses / gap


k = factors(ls)
log_sh, shap_sh, lg_sh, sm_sh = shares(k)
t47a = pd.DataFrame({'LG': k.loc['lg', FACTORS + ['purchases']], 'Samsung': k.loc['samsung', FACTORS + ['purchases']]})
t47a['LG / Samsung'] = t47a['LG'] / t47a['Samsung']
for name, v_ in [('Log share (%)', log_sh), ('Shapley share (%)', shap_sh),
                 ('LG given only this Samsung factor (%)', lg_sh), ('Samsung given only this LG factor (%)', sm_sh),
                 ('Log share, without considerations after an earlier recorded TV purchase (%)', shares(factors(ls[ls['bought_tv_before'] == 0]))[0]),
                 ('Log share, models outside the top models only (%)', shares(factors(rest[rest['brand_group'] != 'other_four']))[0])]:
    t47a[name] = list(v_) + [v_.sum()]
t47a.index = ['Sessions reaching the brand', 'Considerations per session', 'Conversion per consideration (%)', 'Purchased considerations']
t47a.index.name = 'Factor'
save_table('4.7a', 'LG\'s shortfall in purchased TV considerations against Samsung, split into three factors', t47a,
           'Purchased considerations equal sessions times considerations per session times conversion. Each share column splits the '
           'shortfall among the three factors in a different way, and the last row gives the column total. The log share divides the log '
           'of each factor\'s ratio by the log of the ratio of purchased considerations. The Shapley share averages, over the six orders '
           'in which LG\'s three values can be replaced by Samsung\'s, the purchased considerations each replacement adds. The next two '
           'columns replace one factor only, starting from LG or from Samsung, and their totals differ from 100 because the factors '
           'multiply. The last two columns repeat the log share without considerations after an earlier recorded TV purchase and with only the models outside '
           'the top models. This is an accounting identity, with no model.', [F_CONS, F_REST], feeds='Report Figure 4')

fig, ax = plt.subplots(figsize=(7.5, 2.8), layout='constrained')
fac = t47a.iloc[:3]
alt = fac[['Log share (%)', 'Shapley share (%)', 'LG given only this Samsung factor (%)', 'Samsung given only this LG factor (%)']]
yy = np.arange(3)
ax.barh(yy, fac['Log share (%)'], color=['#2F3542', '#C9CDD3', '#8A8D91'], height=0.55)
ax.hlines(yy, alt.min(axis=1), alt.max(axis=1), color='#111111', lw=1.6, zorder=3)
ax.plot(alt.min(axis=1), yy, '|', color='#111111', ms=9, mew=1.6, zorder=3)
ax.plot(alt.max(axis=1), yy, '|', color='#111111', ms=9, mew=1.6, zorder=3)
for i, (sh, rt, hi_) in enumerate(zip(fac['Log share (%)'], fac['LG / Samsung'], alt.max(axis=1))):
    ax.text(max(sh, hi_) + 1.5, i, f'{sh:.0f}%   (LG / Samsung = {rt:.2f})', va='center', fontsize=8)
ax.set_yticks(yy, ['Sessions reaching the brand', 'Considerations per session', 'Conversion per consideration'])
ax.invert_yaxis()
ax.set_xlim(0, 115)
ax.set_xticks(range(0, 101, 20))
ax.xaxis.set_major_formatter(pct)
ax.set_xlabel(f"Share of LG's shortfall in purchased considerations (LG {int(k.loc['lg', 'purchases']):,}, "
              f"Samsung {int(k.loc['samsung', 'purchases']):,})")
ax.grid(axis='y', visible=False)
finish(fig, '4.7a', "LG's shortfall in purchased TV considerations against Samsung, by factor",
       'Bars give the log share from Table 4.7a. The black line spans the lowest and highest share given by the four splits in that '
       'table (log, Shapley and the two one-factor replacements).', [F_CONS], feeds='Report Figure 4')

prod = ls.groupby(['brand', 'product_id']).agg(considerations=('purchased', 'size'), purchases=('purchased', 'sum'),
                                               price=('price', 'median'), top_model=('top_model', 'first')).reset_index()
prod['band'] = pd.cut(prod['price'], BANDS, labels=BAND_NAMES)
pb = prod.groupby(['band', 'brand'], observed=True).agg(models=('product_id', 'size'), considerations=('considerations', 'sum'),
                                                         median_considerations=('considerations', 'median'), purchases=('purchases', 'sum'))
pb['considerations_per_model'] = pb['considerations'] / pb['models']
pb['conversion_pct'] = 100 * pb['purchases'] / pb['considerations']
t47b = pb.unstack('brand')
t47b.columns = [f"{NAMES[br]}, {m.replace('_', ' ').replace('pct', '(%)')}" for m, br in t47b.columns]
t47b.index.name = 'Median price of the model'
save_table('4.7b', 'Considerations per model and conversion by price band, LG and Samsung', t47b,
           f'Each model is placed in the band of its median price at the first view. {UNIT}', [F_CONS], feeds='Report Section 4.7')

tm = {}
for part, d in [('top models', top), ('other models', rest[rest['brand_group'] != 'other_four'])]:
    for br in ['lg', 'samsung']:
        x_ = d[d['brand'] == br]
        allb = ls[ls['brand'] == br]
        tm[f'{NAMES[br]}, {part}'] = {
            'Models': x_['product_id'].nunique(), 'Considerations': len(x_),
            'Share of the brand\'s considerations (%)': 100 * len(x_) / len(allb),
            'Considerations per model': len(x_) / x_['product_id'].nunique(),
            'Sessions that opened one of these models': x_[SESSION].drop_duplicates().shape[0],
            'Purchased considerations': int(x_['purchased'].sum()),
            'Share of the brand\'s purchased considerations (%)': 100 * x_['purchased'].sum() / allb['purchased'].sum(),
            'Conversion (%)': 100 * x_['purchased'].mean(), 'Median price at the first view': x_['price'].median()}
t47c = pd.DataFrame(tm)
t47c.index.name = 'Measure'
gap_cons = 100 * (len(top[top['brand'] == 'samsung']) - len(top[top['brand'] == 'lg'])) / ((ls['brand'] == 'samsung').sum() - (ls['brand'] == 'lg').sum())
gap_buys = 100 * (top.loc[top['brand'] == 'samsung', 'purchased'].sum() - top.loc[top['brand'] == 'lg', 'purchased'].sum()) / (
    ls.loc[ls['brand'] == 'samsung', 'purchased'].sum() - ls.loc[ls['brand'] == 'lg', 'purchased'].sum())
save_table('4.7c', 'Top models against all other models, LG and Samsung', t47c,
           f'A top model holds {TOP_RULE} (Tables 4.7d and 4.7e). The difference between Samsung\'s and LG\'s '
           f'top models is {gap_cons:.0f}% of the brands\' total gap in considerations and {gap_buys:.0f}% of the gap in purchased '
           f'considerations. {UNIT}', [F_TOP, F_REST, F_CONS], feeds='Report Section 4.7')

t47d = cutoffs.T
t47d.index.name = 'Measure'
same_lg = [r for r in cutoffs.index if cutoffs.loc[r, 'LG same models as the chosen rule'] == 'yes' and '(chosen rule)' not in r]
same_sm = [r for r in cutoffs.index if cutoffs.loc[r, 'Samsung same models as the chosen rule'] == 'yes' and '(chosen rule)' not in r]
save_table('4.7d', 'The top-model split under the chosen rule and four alternatives', t47d,
           'Computed by 01_build_tv_dataset.py. Alternatives that select the same LG models as the chosen rule are '
           f"{'; '.join(r.lower() for r in same_lg) or 'none'}. Alternatives that select the same Samsung models are "
           f"{'; '.join(r.lower() for r in same_sm) or 'none'}.", [F_CUT], feeds='Report Section 4.7 and Appendix')

t47e = selection[selection['rank_in_brand'] <= 12].copy()
t47e['brand'] = t47e['brand'].map(NAMES)
t47e['top_model'] = t47e['top_model'].map({1: 'yes', 0: 'no'})
t47e = t47e.set_index(['brand', 'rank_in_brand'])[['product_id', 'considerations', 'share_of_brand_pct', 'cumulative_share_pct',
                                                   'purchased', 'conversion_pct', 'median_price', 'top_model']]
t47e.columns = ['Model', 'Considerations', 'Share of the brand\'s considerations (%)', 'Cumulative share (%)',
                'Purchased considerations', 'Conversion (%)', 'Median price', 'Top model']
t47e.index.names = ['Brand', 'Rank in the brand']
save_table('4.7e', 'The twelve most-considered models of each brand and the top-model rule', t47e,
           'Computed by 01_build_tv_dataset.py, which writes the full list of models.', [F_SEL], feeds='Report Appendix')

prec = []
for name, d in [('All LG and Samsung considerations', ls), ('Top models only', top), ('All models outside the top models', rest),
                ('Without users who bought a TV before', ls[ls['bought_tv_before'] == 0])]:
    d = d[d['brand'].isin(['lg', 'samsung'])]
    gap, se = gap_precision(d)
    p_wild, n_models = wild_p(d)
    pm = d.groupby(['brand', 'product_id'])['purchased'].agg(['mean', 'size']).reset_index()
    big = pm[pm['size'] >= 100]
    prec.append({'Sample': name, 'LG models': int((pm['brand'] == 'lg').sum()), 'Samsung models': int((pm['brand'] == 'samsung').sum()),
                 'LG conversion, weighted by considerations (%)': 100 * d.loc[d['brand'] == 'lg', 'purchased'].mean(),
                 'Samsung conversion, weighted by considerations (%)': 100 * d.loc[d['brand'] == 'samsung', 'purchased'].mean(),
                 'LG minus Samsung (pp)': gap, 'Gap, 95% CI low (pp)': gap - 1.96 * se['user_product'],
                 'Gap, 95% CI high (pp)': gap + 1.96 * se['user_product'],
                 'Standard error, clustered by user and by model (pp)': se['user_product'],
                 'Wild cluster bootstrap p, clustered by model': p_wild,
                 'Standard error, clustered by model only (pp, sensitivity)': se['product'],
                 'Standard error, considerations independent (pp, sensitivity)': se['independent'],
                 'LG average model conversion, all models equally weighted (%)': 100 * pm.loc[pm['brand'] == 'lg', 'mean'].mean(),
                 'Samsung average model conversion, all models equally weighted (%)': 100 * pm.loc[pm['brand'] == 'samsung', 'mean'].mean(),
                 'LG models with 100 or more considerations': int((big['brand'] == 'lg').sum()),
                 'Samsung models with 100 or more considerations': int((big['brand'] == 'samsung').sum()),
                 'LG average model conversion, models with 100 or more considerations equally weighted (%)': 100 * big.loc[big['brand'] == 'lg', 'mean'].mean(),
                 'Samsung average model conversion, models with 100 or more considerations equally weighted (%)': 100 * big.loc[big['brand'] == 'samsung', 'mean'].mean()})
t47f = pd.DataFrame(prec).set_index('Sample')
n_top_models = int(t47f.loc['Top models only', 'LG models'] + t47f.loc['Top models only', 'Samsung models'])
save_table('4.7f', 'How precise the LG minus Samsung conversion gap is, and three ways of weighting the models', t47f,
           'Every consideration of a model carries the same brand, so the comparison rests on the number of models more than on the '
           'number of considerations. The gap, its interval and its main standard error come from a regression of the purchase '
           'indicator on an LG indicator, clustered by user and by model. The wild cluster bootstrap resamples models with Rademacher '
           f'weights and the null imposed, and enumerates every sign pattern when there are 16 models or fewer. The top-model sample has '
           f'{n_top_models} models, so its clustered standard error is unreliable and the bootstrap p-value is the better guide. The three '
           'weightings answer different questions and are not interchangeable, and the last pair covers only the models with at least '
           f'100 considerations. {UNIT}', [F_CONS, F_TOP, F_REST], feeds='Report Section 4.7')

top10 = prod.sort_values('considerations', ascending=False).groupby('brand').head(10)
fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.8), layout='constrained', sharex=True)
xmax = top10['considerations'].max() * 1.3
for n_ax, (ax, br) in enumerate(zip(axes, ['samsung', 'lg'])):
    d = top10[top10['brand'] == br].sort_values('considerations')
    ax.barh(range(len(d)), d['considerations'], color=[COLORS[br] if t == 1 else '#C9CDD3' for t in d['top_model']])
    for i, (vw, pu) in enumerate(zip(d['considerations'], d['purchases'])):
        ax.text(vw + xmax * 0.01, i, f'{100 * pu / vw:.1f}%', va='center', fontsize=7)
    ax.set_yticks(range(len(d)), [f'{p} ({pr_:,.0f})' for p, pr_ in zip(d['product_id'], d['price'])], fontsize=7)
    ax.set_title(f'{chr(65 + n_ax)}. {NAMES[br]}, ten most-considered models')
    ax.set_xlabel('Considerations')
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.grid(axis='y', visible=False)
finish(fig, '4.7b', 'The ten most-considered TV models of Samsung and LG',
       f'Coloured bars are top models, each holding {TOP_RULE}, and grey bars are other models. Labels '
       'give the model ID with its median price in brackets, and the number at the end of each bar is its conversion. Both panels use '
       f'the same scale. {UNIT}', [F_CONS], feeds='Report Section 4.7')

fig, ax = plt.subplots(figsize=(7.5, 3.6), layout='constrained')
sizes = 30 + 220 * t41['LG considerations'] / t41['LG considerations'].max()
ax.scatter(t41['Samsung opened first in two-brand sessions (%)'], t41['LG minus Samsung (pp)'], s=sizes,
           c=[COLORS['lg'] if i == 'TV' else '#8A8D91' for i in t41.index], alpha=0.85, zorder=3)
for name, xv, yv in zip(t41.index, t41['Samsung opened first in two-brand sessions (%)'], t41['LG minus Samsung (pp)']):
    ax.annotate(name, (xv, yv), xytext=(7, 4), textcoords='offset points', fontsize=8)
ax.axhline(0, color='#333333', lw=0.8)
ax.set_ylim(min(-1.2, t41['LG minus Samsung (pp)'].min() - 0.15), max(0.35, t41['LG minus Samsung (pp)'].max() + 0.2))
ax.xaxis.set_major_formatter(pct)
ax.set_xlabel('Share of two-brand sessions in which a Samsung model was opened first')
ax.set_ylabel('LG minus Samsung conversion (pp)')
finish(fig, '4.7c', 'Conversion gap and which brand shoppers open first, by category',
       'Point size is proportional to LG considerations in the category. Six categories only, so the pattern is descriptive. Data in Table 4.1.',
       [F_RAW], feeds='Report Figure 5')

# ---------------------------------------------------------------------------------------------
# Table K. Figures quoted in the text that do not appear in another table
# ---------------------------------------------------------------------------------------------
own_buy = (pe[KEY + ['event_time']].merge(cons[KEY + ['first_view', 'bought_tv_before']], on=KEY)
           .query('event_time >= first_view').groupby(KEY + ['first_view', 'bought_tv_before'])['event_time'].min().reset_index())
own_buy['within_5_min'] = (own_buy['event_time'] - own_buy['first_view']) <= pd.Timedelta(minutes=5)
fast = own_buy.groupby('bought_tv_before')['within_5_min'].mean() * 100
key_figure('Repeat buyers', 'Purchased considerations bought within five minutes of the first view, users who bought a TV before (%)', f'{fast[1]:.1f}', [F_EV, F_CONS])
key_figure('Repeat buyers', 'Purchased considerations bought within five minutes of the first view, all other users (%)', f'{fast[0]:.1f}', [F_EV, F_CONS])
key_figure('Repeat buyers', 'Users who opened a TV after having bought one', f"{cons.loc[cons['bought_tv_before'] == 1, 'user_id'].nunique():,}", [F_CONS])
nbuy = pe.groupby('user_id').size()
key_figure('Repeat buyers', 'Users with more than one TV purchase event, out of all TV buyers', f'{int((nbuy > 1).sum()):,} of {len(nbuy):,}', [F_EV])
key_figure('Repeat buyers', 'Largest number of TV purchase events by one user', f'{int(nbuy.max())}', [F_EV])

pv = events[events['brand'].isin(['lg', 'samsung'])].groupby(['brand', 'product_id'])['price'].agg(['min', 'max', 'nunique'])
for br in ['lg', 'samsung']:
    d = pv.loc[br]
    key_figure('Prices', f'{NAMES[br]} models whose price changed during the two months (%)', f"{100 * (d['nunique'] > 1).mean():.1f}", [F_EV])
    key_figure('Prices', f'{NAMES[br]} models with a price change, median of highest over lowest price minus one (%)',
               f"{100 * (d.loc[d['nunique'] > 1, 'max'] / d.loc[d['nunique'] > 1, 'min'] - 1).median():.1f}", [F_EV])

std_lg = sess[sess['first_group'] == 'lg']
std_sm = sess[sess['first_group'] == 'samsung']
w_band = std_lg['first_band'].value_counts(normalize=True)
sm_rate = std_sm.assign(o4=std_sm['n_o4'] > 0).groupby('first_band', observed=True)['o4'].mean()
key_figure('Where shoppers went', 'Went on to one of the other four brands after a Samsung first TV, reweighted to the price mix of LG first TVs (%)',
           f'{100 * (w_band * sm_rate).sum() / w_band[sm_rate.index].sum():.1f}', [F_CONS])
key_figure('Where shoppers went', 'Went on to one of the other four brands after an LG first TV (%)', f"{100 * (std_lg['n_o4'] > 0).mean():.1f}", [F_CONS])
key_figure('Where shoppers went', 'Sessions that started on Samsung per session that started on LG', f'{len(std_sm) / len(std_lg):.2f}', [F_CONS])
key_figure('Where shoppers went', 'Sessions that started on Samsung and went on to LG', f"{int((std_sm['n_lg'] > 0).sum()):,}", [F_CONS])
key_figure('Where shoppers went', 'Sessions that started on LG and went on to Samsung', f"{int((std_lg['n_sm'] > 0).sum()):,}", [F_CONS])

key_figure('Top models', 'No LG or Samsung model holds a share of its brand\'s considerations between these values, so any cutoff in '
           'the range gives the same split (%)',
           f"{selection.loc[selection['top_model'] == 0, 'share_of_brand_pct'].max():.2f} to "
           f"{selection.loc[selection['top_model'] == 1, 'share_of_brand_pct'].min():.2f}", [F_SEL])
key_figure('Top models', 'LG models with a median price above 650, out of all LG models',
           f"{int(((prod['brand'] == 'lg') & (prod['price'] > 650)).sum())} of {int((prod['brand'] == 'lg').sum())}", [F_CONS])

key_figure('Reviews file', 'Review rows for TVs', f"{int(reviews['product_category'].str.lower().str.contains('tv').sum())}", [F_REV])
key_figure('Reviews file', 'Product categories in the reviews file', ', '.join(sorted(reviews['product_category'].unique())), [F_REV])
key_figure('Reviews file', 'Review product IDs that also appear in the TV data', f"{len(set(reviews['product_id']) & set(cons['product_id']))}", [F_REV, F_CONS])

tk = pd.DataFrame(KEYFIG)
tk.index = [f'K{i + 1}' for i in range(len(tk))]
tk.index.name = 'Item'
save_table('K', 'Figures quoted in the text that do not appear in another table', tk,
           'Each value is computed in this script from the file named in its row.', sorted(set(sum([r['Built from'].split(', ') for r in KEYFIG], []))),
           feeds='Report text')

# ---------------------------------------------------------------------------------------------
# HTML overview of every exhibit, with an index of the files each one is built from
# ---------------------------------------------------------------------------------------------
css = ('body{font-family:Inter,Arial,sans-serif;font-size:13px;color:#1d1d1f;max-width:1100px;margin:24px auto;padding:0 16px}'
       'h1{font-size:20px}h2{font-size:15px;margin:28px 0 6px}p.note{color:#555;font-size:12px;margin:4px 0 6px}'
       'p.feeds{color:#1428A0;font-size:12px;margin:0 0 8px}table{border-collapse:collapse;font-size:11.5px;margin:6px 0 4px}'
       'th,td{border:1px solid #d0d3d8;padding:3px 7px;text-align:right;vertical-align:top}th{background:#f0f2f5}'
       'td:first-child,th:first-child{text-align:left}img{max-width:100%;max-height:170mm;border:1px solid #eee}'
       'table.index td{text-align:left}'
       'section{break-inside:avoid;page-break-inside:avoid;margin-bottom:22px}h2{break-after:avoid}'
       '@page{size:A4 landscape;margin:12mm}')


def fmt_cell(v):
    if isinstance(v, (int, np.integer)):
        return f'{v:,}'
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return ''
        return f'{v:,.0f}' if float(v).is_integer() and abs(v) >= 10 else f'{v:,.2f}'
    return v


idx_rows = ''.join(f"<tr><td>{e['kind']} {e['id']}</td><td>{html.escape(e['title'])}</td><td>{html.escape(', '.join(e['src']))}</td></tr>"
                   for e in EXHIBITS)
parts = [f'<!doctype html><html><head><meta charset="utf-8"><title>Preliminary analysis, Sections 1 to 4</title><style>{css}</style></head><body>',
         '<h1>Preliminary analysis for report Sections 1 to 4 (descriptive, no models)</h1>',
         f'<p class="note">Generated by 02_preliminary_analysis.py on {pd.Timestamp.now():%d %B %Y}. Data from the retailer behaviour log, '
         f'1 October to 30 November 2019, with exact duplicate events removed. {html.escape(UNIT)}</p>',
         '<div><h2>Index of exhibits and the files they are built from</h2><table class="index"><tr><th>Exhibit</th><th>Title</th>'
         f'<th>Built from</th></tr>{idx_rows}</table></div>']
for ex in EXHIBITS:
    parts.append('<section>')
    parts.append(f"<h2>{ex['kind']} {ex['id']}. {html.escape(ex['title'])}</h2>")
    if ex['feeds']:
        parts.append(f"<p class='feeds'>Feeds {html.escape(ex['feeds'])}</p>")
    if ex['kind'] == 'Table':
        parts.append(ex['df'].map(fmt_cell).to_html())
        parts.append(f"<p class='note'>Note. {html.escape(ex['note'])}</p>")
    else:  # the figure carries its title, note and source file
        parts.append(f"<img src='figures/{ex['path'].name}' alt='{html.escape(ex['title'])}'>")
    parts.append('</section>')
parts.append('</body></html>')
(OUT / 'preliminary_analysis.html').write_text('\n'.join(parts), encoding='utf-8')
pd.DataFrame([{'Exhibit': f"{e['kind']} {e['id']}", 'Title': e['title'], 'Built from': ', '.join(e['src'])} for e in EXHIBITS]).to_csv(
    OUT / 'exhibit_index.csv', index=False)
print(f'\nWrote {sum(e["kind"] == "Table" for e in EXHIBITS)} tables and {sum(e["kind"] == "Figure" for e in EXHIBITS)} figures to {OUT}')
