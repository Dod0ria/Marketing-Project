# Review of proposed step 03

**Verdict: approved with reservations, 7/10.** The analysis is reproducible and has a useful connection to the reach/conversion question. The main remaining problems concern the definition of the choice set, alignment of estimands and interpretation of inference. They do not establish that the reported coefficients were fabricated or incorrectly calculated.

Source: user-supplied `/Users/dodo/Downloads/03_models.html`, an executed notebook export, not yet an executable file in this repository. Section and input-cell references below refer to this export. `00_scope_and_methodology_rationale.ipynb` now describes its proposed role, inputs, four model blocks and outputs. Its roadmap explicitly distinguishes the proposal from a committed notebook.

## What was checked

The HTML's displayed Python was inspected and extracted. Current `01_build_tv_dataset.py` was run against the original CSV in a separate temporary workspace. Its derived files were used to execute the submitted model code, including the price extension. No existing project analysis, raw dataset or supplied HTML was edited.

Reproduced results include:

- Model 1 baseline LG coefficient: -0.5167 pp, two-way SE 0.2905, wild product-cluster p about 0.124; extended coefficient -0.5214 pp.
- Conditional choice Logit: LG odds ratio 0.7823, reported covariance p about 0.0878. The notebook's comparison of coefficients with statsmodels passes.
- Product-level Poisson: LG coefficient -0.8904, equivalent to approximately 58.95% fewer considerations at the same model-level median price under the specified conditional-mean model.
- Daily price extension: a 10% higher price is associated with about 22.52% fewer considerations; the daily and weekly Poisson fits converged. Convergence is not validation of the exposure window, imputed prices or identifying assumptions.

The review adds targeted checks on purchase-time eligibility and LPM probability ranges. Temporary diagnostics are in `/private/tmp/marketing_03_review/`; they are not new official pipeline outputs.

## P1 — Correct before using the results as final evidence

### 1. Close the choice set at the purchase time

**Location:** Model 2, Section 4, input cell 7. **Type:** sample-construction error relative to the claimed choice interpretation.

`choice_visits` selects visits with at least two considerations and one purchased consideration. `ch` then retains every considered product in the whole visit. It does not require an alternative's first view to precede the purchase.

The reproduced sample has 4,302 alternatives in 1,265 visits. **150 alternatives in 89 visits (7.04% of choice visits) were first opened strictly after the chosen purchase.** They cannot be described as products already opened when that purchase was made.

**Fix:** obtain the chosen product's qualifying purchase timestamp from `tv_events_clean.csv`; keep only alternatives first viewed at or before it, with same-second ambiguity documented; then reapply the minimum-choice-set rule and recompute head-to-head membership. Check additional purchase events, including purchases not represented by a qualifying consideration. One selected visit contains purchase events for another product. Multiple purchased considerations identify multiple distinct products, not physical quantities.

A targeted sensitivity that only removes late alternatives and reapplies the two-option requirement leaves **4,108 alternatives in 1,221 visits**. LG's OR changes from 0.7823 to 0.7813 with the submitted covariance calculation. The near-identical coefficient is reassuring, but does not remove the need to correct the sample. This sensitivity has not resolved the additional-purchase ambiguity or validated the custom covariance.

### 2. Separate brand reach from considerations per product

**Location:** introduction, Sections 5 and 7. **Type:** mismatch between estimands and narrative.

The decomposition in `02` uses brand-reaching user-sessions, considerations per reaching session and conversion. Model 3 instead regresses **considerations per product**. For a single product, consideration count equals visits opening that product; across a brand, summing these counts counts a visit repeatedly if several models were opened. It combines reach, breadth of browsing and the observed assortment. The Poisson coefficient is not a re-estimation or statistical confirmation of the approximately 70% accounting contribution.

The claim that log-count OLS describes a typical model while Poisson describes total brand traffic is also too strong. Their key distinction is the conditional mean of log counts versus an exponential conditional mean of counts. Poisson's coefficient still describes a per-product conditional-mean comparison. Robust errors accommodate variance misspecification under appropriate conditions; they do not rescue an incorrect conditional mean, omitted product characteristics or dependence between products sharing shoppers.

**Fix:** consistently call Model 3 an adjusted comparison of *considerations per observed product*. Explain its relation to the identity `brand considerations = brand-reaching sessions × considerations per reaching session`, and to assortment size, without claiming the same endpoint. Describe the OLS/Poisson difference as sensitivity to the scale and mean specification; inspect price support and influential products. The population contains observed viewed models, not a known full catalogue including never-viewed products.

### 3. Use a declared inference convention and validate custom uncertainty

**Location:** helpers, Tables 5.1/5.3/5.4. **Type:** inferential comparability and validation gap, not a demonstrated error in the reproduced point estimates.

`ols2` uses user-and-product clustering. `wild_p` uses product-level signs and a product-cluster statistic. These do not impose the same dependence structure. In Table 5.3, several two-way intervals exclude zero while the product-wild p-values exceed .05; for example, the price-band row has clustered p about .030 versus wild p about .083. Neither result can be silently substituted for the other. The within-visit LPM also has p about .032, although the conditional Logit has p about .088: avoid a global claim that all adjusted comparisons are statistically indistinguishable from zero.

The conditional Logit checks its coefficients against statsmodels, but that does not independently validate the hand-built two-way covariance. The product allocation of centred choice-set scores requires justification. The optimiser's `.x` is used without checking success or the final gradient. Enumerating all 8,192 signs for 13 products removes bootstrap Monte Carlo error; it does not make inference automatically exact or reliable.

**Fix:** state the primary inferential procedure and the assumptions behind each sensitivity before writing a significance-based conclusion. Label the product-only bootstrap explicitly. Add optimisation diagnostics and an independent covariance check or justified resampling procedure for Model 2. For the product-count model, HC1 is not a check of shared-user dependence across products. Do not select the procedure because it produces a preferred p-value.

Relevant primary references: [statsmodels two-group covariance](https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_cluster_2groups.html), [ConditionalLogit group scores](https://www.statsmodels.org/dev/generated/statsmodels.discrete.conditional_models.ConditionalLogit.score_grp.html), [MacKinnon, Nielsen and Webb, cluster-robust inference](https://arxiv.org/abs/2205.03285).

## P2 — Important improvements to interpretation and robustness

### 4. Replace “same shopper” and causal-sounding explained shares

**Location:** Model 1 title, Gelbach discussion, final synthesis. **Type:** interpretation.

Model 1 compares rows after adjustment for observed state, price and week. It does not hold the individual shopper, specifications, stock or intent fixed. Model 2 conditions within a visit, but only among selected multi-option, single-purchase visits; it does not identify unconditional conversion or a causal brand effect. Visit fixed effects absorb common visit-level components, not brand-specific preferences or all omitted alternative attributes. The opened set is an observed consideration set, not an externally assigned choice set.

**Fix:** use “conditional on observed browsing state, price and week”. Describe Gelbach's 37% as an accounting contribution to the change in the fitted LG coefficient, not a causal fraction of the commercial gap. The coefficient-change identity and displayed contributions reproduce; the concern is interpretation and missing uncertainty on the components. “Price explains nothing” should become “the price contribution is near zero in this additive specification”. Calendar-week effects absorb common weekly shifts, not all product-specific promotions.

The 44% worked example also assumes exactly two options, neither opened first, which conflicts with the observed definition of a first TV in a two-option visit. Present it as a hypothetical equal-covariate odds comparison, or construct a coherent scenario and include all relevant option attributes and their uncertainty. It is not LG's observed purchase conversion.

### 5. Keep the within-product price analysis an optional, qualified extension

**Location:** Model 3 observed-span offset; Model 4, Section 6/input cell 9. **Type:** measurement and identification limits.

First-to-last recorded event is an **observed activity span**, not known listing or availability. Unobserved days can mean no sampled traffic, unavailability or missing sessions. The daily panel has 9,405 model-days, with about 32.23% containing no consideration. Zero consideration does not imply zero events, but prices in genuinely event-free periods are imputed with forward/back filling; these are not observed posted prices. Price is aggregated over view/cart/purchase events, so the recorded mix can also affect the daily median. A weekly specification is useful but does not independently resolve these shared assumptions.

**Fix:** rename `days_listed`; quantify periods with imputed versus observed prices; compare defensible window and price-observation rules and report how results change. Keep no causal elasticity, traffic lift or discount recommendation. Without those checks, omit Model 4 from the central argument rather than expand it. “No detectable conversion association” does not establish that price operates only through traffic.

### 6. Report the LPM diagnostic instead of rejecting or overlooking it

**Location:** Model 1/input cells 3 and 5. **Type:** model diagnostic.

Reproduced fitted probabilities are outside [0,1] for **0.40%** of baseline M2 rows and **0.94%** of extended M3 rows, all below zero. The minima are approximately -0.73% and -1.24%. This does not by itself invalidate an average LPM contrast; the matching Logit results are a real strength. It does limit individual probability interpretations and merits a short diagnostic.

**Fix:** display these fractions, keep the Logit average discrete contrasts and their intervals, and verify common support across brand, price and browsing state. Do not clip LPM predictions to manufacture valid probabilities.

### 7. Correct timing, sample and prespecification statements

**Location:** conventions, robustness notes, top-model interaction. **Type:** documentation with analytical implications.

- `top_model` is based on the whole two-month period, so the statement that every control was known at first view needs an explicit retrospective-split exception.
- `bought_tv_before == 0` removes flagged **considerations**, not every observation belonging to users who ever purchased before. In the LG/Samsung sample the flag marks 743 rows belonging to 394 users. Rename the check; do not silently broaden the exclusion.
- Starting on 8 October allows seven elapsed days in the file, not complete seven-day user histories; sessions were sampled.
- The claim that specifications were fixed before any model was run is not established by the HTML. Prior exploration already exists. Retain only if the team can document that chronology; otherwise describe the actual exploratory development.
- Conditional Logit's substitution assumptions and selected single-purchase sample need discussion. There is no obligation to add another choice model merely for complexity.

**Fix:** correct these descriptions before interpreting results and keep a short record of actual specification decisions.

## P3 — Packaging and project economy

Upload the executable `.ipynb`, with dependencies and reproducible inputs, rather than only its HTML export. Keep the HTML as an optional read-only rendering. Preserve Tables 5.1–5.6 and the exhibit index as aggregate outputs, without treating all of them as necessary report exhibits. Verify that advanced methods such as Gelbach, conditional choice Logit and Poisson match the methods taught or have appropriate instructor clearance; the supplied course requirements do not imply that every advanced extension is required.

Recommended order: **correct Model 2's choice timing → align the reach endpoint and interpretations → settle inference and validate custom code → revise the price extension and diagnostics → simplify the report and upload the notebook.** The model code runs; the project does not need to start again.
