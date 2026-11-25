# Data dictionary for the LG TV project

`01_build_tv_dataset.py` writes two files to `../data/01_LG_Electronics_Appliances/`. All times are in UTC.

`events_clean.csv` is the course behaviour file with its exact duplicate events removed. It keeps all six product categories and the original eight columns, sorted by time, with event type, brand and category in lower case.

`tv_considerations.csv` has one row per user, session and TV model. Load it with `pd.read_csv(path, parse_dates=['first_view'])`.

## The unit of analysis

A row is a consideration, meaning one user, in one session, opening one TV model. It is timed at the first view of that model in that session, and its price is the price recorded at that view. A session is identified by the user and session IDs together, because a few session IDs appear under more than one user. A consideration counts an opportunity to buy. One user can open many models and one model can be viewed many times, so considerations count neither people nor units sold.

## Columns

The column "Known at the first view" says whether the value was available when the shopper first opened the model. Only those columns can describe the shopper's situation in a model of the purchase.

| Column | Known at the first view | Definition |
| --- | --- | --- |
| `user_id`, `session_id`, `product_id` | yes | Identify the row |
| `brand` | yes | artel, haier, kivi, lg, samsung or tcl |
| `brand_group` | yes | lg, samsung or other_four (Artel, Haier, Kivi and TCL together) |
| `first_view` | yes | Time of the first view of this model in this session |
| `month` | yes | 2019-10 or 2019-11, from `first_view` |
| `week` | yes | ISO calendar week of `first_view`, from 40 (which starts on Tuesday 1 October in the data) to 48 (25 to 30 November) |
| `price` | yes | Price at the first view, in the retailer's unidentified currency |
| `tvs_opened_before` | yes | Number of other TV models opened earlier in the session |
| `lg_opened_before` | yes | 1 when an LG TV was opened earlier in the session. For an LG row this means another LG model. |
| `samsung_opened_before` | yes | 1 when a Samsung TV was opened earlier in the session. For a Samsung row this means another Samsung model. |
| `other_four_opened_before` | yes | 1 when a TV of Artel, Haier, Kivi or TCL was opened earlier in the session. For a row of one of these brands it can be another model of the same brand. |
| `other_brand_opened_before` | yes | 1 when a TV of any brand other than this row's brand was opened earlier in the session |
| `rival_opened_before` | yes | LG and Samsung rows only. 1 when the other main brand was opened earlier in the session (Samsung for an LG row, LG for a Samsung row). Empty for the other four brands. |
| `own_other_model_opened_before` | yes | LG and Samsung rows only. 1 when another model of the row's own brand was opened earlier in the session. Empty for the other four brands. |
| `position_in_session` | yes | first_tv (no TV opened earlier in the session), after_same_brand_only (only models of the same brand opened earlier) or after_other_brand (a TV of another brand opened earlier) |
| `bought_tv_before` | yes | 1 when the same user bought a TV before this row, in this or an earlier session in the data |
| `returning_visitor` | yes | 1 when the user had an earlier session in the data, in any category |
| `model_share_of_brand_pct` | no, retrospective | LG and Samsung rows only. The model's share of its brand's considerations over the two months (%). Empty for the other four brands. |
| `model_rank_in_brand` | no, retrospective | LG and Samsung rows only. The model's rank within its brand by considerations, 1 being the most considered, with ties ordered by product ID. Empty for the other four brands. |
| `top_model` | no, retrospective | 1 when `model_share_of_brand_pct` is at least 4%. Always 0 for the other four brands. |
| `purchased` | no, outcome | 1 when this model is bought at or after the first view, in the same session. This is the main outcome. |
| `added_to_cart` | no, outcome | 1 when this model is added to the cart at or after the first view, in the same session |
| `purchased_after_cart` | no, outcome | 1 when this model is bought after its first cart event |
| `session_result` | no, outcome | What was bought after this model was opened, in the same session (see below) |

`lg_opened_before` and `samsung_opened_before` mean different things for the two main brands, because each one refers to the row's own brand on half of the rows. `rival_opened_before` and `own_other_model_opened_before` describe the same history in the same terms for LG and Samsung, which is why the models use them.

## Measurement rules

**Outcome window.** A purchase or a cart addition counts when it concerns this model, happens in the same session and is logged at or after the first view. Events are recorded to the second, so an event logged in the same second as the first view counts. A purchase of the same model in a later session does not count, and Table 2.2 of the descriptive analysis shows how often that happens.

**History within the session.** "Earlier in the session" covers the whole session before this model was opened, because the history is meant to describe the search in progress. A TV purchase earlier in the same session does not restart it. Table 2.4 of the descriptive analysis shows the alternative definition that does.

**All six TV brands.** The other four brands stay in the data because they are part of the paths that lead shoppers to LG and Samsung models. Grouping them as `other_four` is a convenience and does not imply that they are positioned alike.

**Earlier activity of the user.** `returning_visitor` looks at the user's earlier sessions. `bought_tv_before` looks at the user's earlier sessions and also at the part of the current session before the first view, so a TV bought earlier in the same session sets it. They are the only columns that use the user's activity outside the current session. The course file is a sample of sessions, so a user's earlier visits and purchases can be missing. Both flags can therefore miss earlier activity, but they never record activity that did not happen.

**Model popularity.** `model_share_of_brand_pct`, `model_rank_in_brand` and `top_model` are computed over the whole two months. They describe how popular a model was over the period and say nothing about what the shopper could see at the first view, so they are used to split the sample and never as a description of the shopper. Table 2.13 of the descriptive analysis compares the 4% cutoff with three alternative rules.

**Controls.** The controls in the models of the purchase are columns marked yes in "Known at the first view", with one exception. `top_model` enters Model 1c through its interaction with LG and enters Model 2 as a covariate. Those two specifications therefore describe purchases using information that was not available at the first view, and they cannot predict a purchase at that moment. Outcome columns, including `session_result`, never enter a model as controls.

## session_result

`session_result` takes one value per row, by priority. If this model was bought, it is bought_this_model. Otherwise, if a TV of another brand was bought later in the session, it is bought_lg, bought_samsung or bought_other_four, for the first such purchase. Otherwise, if another model of the same brand was bought, it is bought_other_model_same_brand. Otherwise it is bought_nothing.

The column classifies each consideration. A session with several considerations can therefore count one purchase more than once, so statements about which brand a session ended up buying rely on Table 2.8 of the descriptive analysis, which counts each session once.
