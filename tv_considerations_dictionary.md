# Working datasets for the LG TV project

All files are built by `01_build_tv_dataset.py` from `electronics_market_behavior.csv` (1 October to 30 November 2019) and saved in `../data/01_LG_Electronics_Appliances/`, outside the repository. All times are in UTC. The course describes the behaviour file as a curated sample, so a user's earlier sessions can be missing from it.

| File | What it holds |
| --- | --- |
| `tv_events_clean.csv` | The TV events of the six TV brands after exact duplicates are removed, with the original eight columns, sorted by time |
| `tv_considerations.csv` | One row per user, session and TV model |
| `tv_considerations_top_models.csv` | The rows of LG's and Samsung's top models only |
| `tv_considerations_without_top_models.csv` | Every other row, meaning the other LG and Samsung models and all rows of the other four brands |

The two split files have the same columns as `tv_considerations.csv` and together contain exactly its rows. Their variables are computed on full sessions before the split, so a Samsung top model opened before an LG model still counts as a Samsung TV opened earlier.

The build script also writes these tables to `results/build/`.

| File | What it shows |
| --- | --- |
| `build_parameters.csv` | The top-model cutoff, the price-similarity range and the history-check start date, read by the analysis script |
| `data_audit.csv` | Exclusions, key uniqueness, event-order checks, same-second events and session lengths |
| `top_model_selection.csv` | Every LG and Samsung model ranked by considerations, with its share of the brand, cumulative share, conversion and whether it is a top model |
| `top_model_cutoffs.csv` | The top-model split under the chosen rule and four alternatives |
| `variable_decisions.csv` | Conversion by each variable kept in the dataset and by each browsing variant left out |
| `reset_sensitivity.csv` | Position in the session over the whole session and since the last TV purchase in the session |

## tv_considerations.csv

A row (a consideration) is a user, in one session, opening one TV model, timed at the first view of that model in that session. Load it with `pd.read_csv(path, parse_dates=['first_view'])`.

The column "Known at the first view" says whether the value was available when the shopper first opened the model. Only those columns can describe the shopper's situation in a model of the purchase. The others are outcomes.

| Column | Known at the first view | Definition |
| --- | --- | --- |
| `user_id`, `session_id`, `product_id` | yes | Identify the row. A session is a user and session pair, because four session IDs appear under more than one user. |
| `brand` | yes | artel, haier, kivi, lg, samsung or tcl |
| `brand_group` | yes | lg, samsung or other_four (Artel, Haier, Kivi and TCL together) |
| `top_model` | yes, from the two months as a whole | 1 when the model holds at least 4% of its brand's considerations over the two months. Only LG and Samsung models can be top models. |
| `first_view` | yes | Time of the first view of this model in this session |
| `month` | yes | 2019-10 or 2019-11, from `first_view` |
| `price` | yes | Price at the first view, in the retailer's unidentified currency |
| `tvs_opened_before` | yes | Number of other TV models opened earlier in the session |
| `lg_opened_before` | yes | 1 when an LG TV was opened earlier in the session. For an LG row this means another LG model. |
| `samsung_opened_before` | yes | 1 when a Samsung TV was opened earlier in the session. For a Samsung row this means another Samsung model. |
| `other_four_opened_before` | yes | 1 when a TV of Artel, Haier, Kivi or TCL was opened earlier in the session. For a row of one of these brands it can be another model of the same brand. |
| `other_brand_opened_before` | yes | 1 when a TV of any brand other than this row's brand was opened earlier in the session |
| `position_in_session` | yes | first_tv (no TV opened earlier in the session), after_same_brand_only (only models of the same brand opened earlier) or after_other_brand (a TV of another brand opened earlier) |
| `bought_tv_before` | yes | 1 when the same user bought a TV before this row, in this or an earlier session in the data |
| `returning_visitor` | yes | 1 when the user had an earlier session in the data, in any category |
| `purchased` | no, outcome | 1 when this model is bought at or after the first view, in the same session (main outcome) |
| `added_to_cart` | no, outcome | 1 when this model is added to the cart at or after the first view, in the same session |
| `purchased_after_cart` | no, outcome | 1 when this model is bought after its first cart event |
| `session_result` | no, outcome | What was bought after this model was opened, in the same session (see below) |

"Earlier in the session" covers the whole session before this model was opened. A TV purchase earlier in the same session does not restart the history. `reset_sensitivity.csv` shows the alternative that does, which changes at most 364 considerations, all of them flagged by `bought_tv_before`.

`top_model` uses the full two months, so it describes the model's popularity over the period, not what the shopper could see at the first view. Use it to split the sample, not as a shopper characteristic.

A cart or purchase logged in the same second as the first view counts as at or after it. `data_audit.csv` gives the counts.

### session_result

`session_result` takes one value per row, by priority. If this model was bought, it is bought_this_model. Otherwise, if a TV of another brand was bought later in the session, it is bought_lg, bought_samsung or bought_other_four, for the first such purchase. Otherwise, if another model of the same brand was bought, it is bought_other_model_same_brand. Otherwise it is bought_nothing.

It classifies the consideration and is not a list of every purchase in the session. A session with several considerations can count one purchase more than once. Claims about which brand a session ended up buying use the session-level table in the analysis (Table 4.5d), which counts each session once.

### Outside the session

`bought_tv_before` and `returning_visitor` are the only columns that use earlier sessions. Because the data are a sample, earlier sessions can be missing, so both flags can miss earlier visits and purchases but never invent them. The 1,256 rows with `bought_tv_before` = 1 convert at 26.5%, against 2.8% for the rest, and the analysis reports its main results with and without them.

## Browsing variants left out

The earlier version also measured whether another brand had been viewed in earlier sessions (within 48 hours, within 7 days or at any time), and whether the other brand opened earlier was similarly priced. These were dropped because the course file samples sessions, so earlier sessions are incomplete, and because the question concerns the search in progress. `variable_decisions.csv` reports how they relate to conversion, and they can be rebuilt from `tv_events_clean.csv` for sensitivity checks in the models.

## Checks

`03_verify_variables.py` recomputes every derived column for a stratified sample of rows drawn with seed 2026, using plain loops over each user's events that share no code with the build script. The sample includes every special case. It also runs logical checks on the full file, checks that the two split files contain exactly the rows of the main file, and recomputes the top-model rule from the events. Its output is saved in `results/03_verification_output.txt`.
