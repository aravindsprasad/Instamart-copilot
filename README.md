# Instamart Customer Intelligence & Support Copilot

## Overview

A synthetic-data capstone in four connected Parts, built with the Python standard library, pandas, numpy, scikit-learn and joblib only.
Every language-model call goes through the supplied offline `mock_llm.py`; no network, API key or paid service is used anywhere.
All data is synthetic and fictional; all money is in INR.

| Part | What it does | Main files |
|---|---|---|
| A | Audit, clean and store the raw exports; SQL and pandas analysis | `src/a1_audit_clean.py`, `src/a2_sql_analysis.py`, `src/a3_validate.py`, `sql/queries.sql` |
| B | RFM + K-means segments; lapse model (leak-free pipeline) | `src/b1_rfm_segments.py`, `src/b2_lapse_model.py` |
| C | Ticket triage with guardrails, retries, validation, budget and routing | `src/c_triage.py` |
| D | TF-IDF policy RAG assistant and a tool-using, resumable refund agent | `src/rag.py`, `src/agent.py`, `src/d1_rag_eval.py`, `src/d2_agent_run.py` |

`generate_data.py` and `mock_llm.py` are the supplied files, committed unmodified. Each script prints its results and saves them as
`reports/*.md`; `build_readme.py` assembles this README from those fragments, so every number below was produced by the code.

## Reproduce

Run from the repository root (Python 3.11 or newer; SQLite 3.25 or newer). The commands are listed in order.

1. `python -m venv .venv`
2. `source .venv/bin/activate` (on Windows: `.venv\Scripts\activate`)
3. `pip install -r requirements.txt`
4. `python generate_data.py` (last line must be `generated: customers=1200 products=40 orders_raw=13084 tickets=164`)
5. `python src/a1_audit_clean.py` (audit, cleaning, `data/clean/*.csv`, `data/instamart.db`)
6. `python src/a2_sql_analysis.py` (runs the seven queries of `sql/queries.sql`, pandas cross-check)
7. `python src/a3_validate.py` (prints `ALL CHECKPOINTS PASSED`)
8. `python src/b1_rfm_segments.py` (writes `outputs/customer_segments.csv`)
9. `python src/b2_lapse_model.py` (writes `outputs/lapse_model.joblib`, `outputs/retention_list.csv`)
10. `python src/c_triage.py` (writes `outputs/triage_budget_demo.csv`, `outputs/triage_results.csv`, `outputs/retry_log.txt`)
11. `python src/d1_rag_eval.py` (chunking, retrieval and grounding evaluation)
12. `python src/d2_agent_run.py` (writes `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`, `state/`)
13. `python build_readme.py` (optional: rebuilds this README from `reports/*.md`)

Notes: random seeds are fixed (K-means 42, split 42, forest 42, resampling 42, retry jitter 42). Steps 5 to 12 stop with an assertion error if a reference checkpoint is missed.
`requirements.txt` pins the library versions the project was developed and tested with (Python 3.12).

## Part A

### A.1 Audit of the raw data

**orders.csv** - shape 13,084 rows x 9 columns

| column | dtype | missing | missing_% |
|---|---|---|---|
| order_id | str | 0 | 0.00 |
| customer_id | str | 0 | 0.00 |
| product_id | str | 0 | 0.00 |
| quantity | int64 | 0 | 0.00 |
| order_ts | str | 0 | 0.00 |
| status | str | 0 | 0.00 |
| discount_pct | int64 | 0 | 0.00 |
| order_amount | str | 53 | 0.41 |
| delivery_minutes | float64 | 2541 | 19.42 |

**customers.csv** - shape 1,200 rows x 5 columns

| column | dtype | missing | missing_% |
|---|---|---|---|
| customer_id | str | 0 | 0.00 |
| city | str | 0 | 0.00 |
| signup_date | str | 0 | 0.00 |
| age | float64 | 61 | 5.08 |
| gender | str | 30 | 2.50 |

**products.csv** - shape 40 rows x 4 columns

| column | dtype | missing | missing_% |
|---|---|---|---|
| product_id | str | 0 | 0.00 |
| product_name | str | 0 | 0.00 |
| category | str | 0 | 0.00 |
| price_inr | int64 | 0 | 0.00 |

_Note: `pd.read_csv` treats the literal text `N/A` as missing, so the `order_amount` missing count above includes the `N/A` entries; the exact non-numeric count below is taken from an all-string read._

**Audit figures**

| audit figure | value | reference | match |
|---|---|---|---|
| orders.csv rows | 13,084 | 13,084 | yes |
| exact duplicate rows | 119 | 119 | yes |
| rows with status pending | 656 | 656 | yes |
| distinct order_id | 12,309 | 12,309 | yes |
| order_amount not convertible to a number | 120 | 120 | yes |
| negative delivery_minutes | 106 | 106 | yes |
| blank delivery_minutes | 2,541 | 2,541 | yes |
| customers: blank age | 61 | 61 | yes |
| customers: age outside 10-100 | 6 | 6 | yes |
| customers: distinct raw city spellings | 24 | 24 | yes |
| customers: distinct non-blank raw gender spellings | 10 | 10 | yes |
| customers: blank gender | 30 | 30 | yes |

### A.2 Cleaning rules - customers

| rule | values changed |
|---|---|
| (a) city: strip + title case -> 6 cities | 476 city strings |
| (b) gender: map to male / female / unknown (blank -> unknown) | 950 gender strings rewritten (30 blank -> unknown) |
| (c) ages outside 10-100 set to missing | 6 ages |
| (c) missing ages (blank + invalid) filled with city median (groupby + transform) | 67 ages (61 blank + 6 invalid) |
| (d) signup_date -> datetime | 1200 dates |

City counts after cleaning: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184

Gender counts after cleaning: female 610, male 560, unknown 30

Missing ages: 0; age range 18-60.

### A.3 Cleaning rules - orders (applied in this order)

| rule | values/rows changed | rows after |
|---|---|---|
| 1. order_ts -> datetime | 13,084 values converted | 13,084 |
| 2. drop exact duplicate rows | 119 rows removed | 12,965 |
| 3. keep latest order_ts per order_id | 656 rows removed (earlier 'pending' records) | 12,309 |
| 4. order_amount -> numeric, fill from price x qty x (1 - disc/100) | 112 values filled | 12,309 |
| 5. negative delivery_minutes -> missing; completed filled with category median; cancelled left missing | 105 negatives set missing; 774 completed values filled; 1199 cancelled left missing | 12,309 |

| check | value | reference | match |
|---|---|---|---|
| rows | 12,309 | 12,309 | yes |
| completed | 11,110 | 11,110 | yes |
| cancelled | 1,199 | 1,199 | yes |
| pending | 0 | 0 | yes |
| missing amount | 0 | 0 | yes |
| missing delivery (completed) | 0 | 0 | yes |
| missing delivery (cancelled) | 1,199 | 1,199 | yes |
| completed revenue INR | 2,998,668.80 | 2,998,668.80 | yes |

### A.4 Storage

Wrote `data/clean/customers.csv` (1,200 rows), `products.csv` (40 rows), `orders.csv` (12,309 rows) and the same three tables to `data/instamart.db` (`order_ts` stored as text `YYYY-MM-DD HH:MM:SS`).

### A.5 SQL analysis (queries in `sql/queries.sql`)

| query | result | reference |
|---|---|---|
| Q1 WHERE then HAVING | 30 customers | 30 |
| Q2 three-table join | 30 rows; sum INR 2,998,668.80; largest Bengaluru / Electronics INR 233,909.45 | 30 rows; INR 2,998,668.80; Bengaluru / Electronics INR 233,909.45 |
| Q3 scalar subquery | 11 products | 11 |
| Q4 LEFT JOIN | 25 customers without a completed order | 25 |
| Q5 CTE | 550 customers above average INR 2,552.06 | 550; INR 2,552.06 |
| Q6 window ranking | 1175 rows; 8 rows with RANK != ROW_NUMBER | 8 |
| Q7 running total | 183 days; last cumulative INR 2,998,668.80 | 183; INR 2,998,668.80 |

**Q6 top customer per city:** Bengaluru C0082 (12,856.20); Chennai C0930 (10,151.40); Delhi C0291 (13,742.60); Hyderabad C0743 (12,730.05); Mumbai C0693 (14,766.55); Pune C0426 (10,377.10)

Why RANK() and ROW_NUMBER() differ: when two customers in a city have exactly the same total spend, RANK() gives them the same rank (and skips the next number) while ROW_NUMBER() still hands out a unique, arbitrary sequence number, so from the second tied row onwards the two columns disagree.

### A.6 pandas cross-check and exploratory analysis

**Pivot table of completed revenue (INR), city x category, with margins**

| city | Dairy | Electronics | Grocery | Personal Care | Snacks | Total |
|---|---|---|---|---|---|---|
| Bengaluru | 87,075.05 | 233,909.45 | 110,915.65 | 20,193.85 | 71,549.40 | 523,643.40 |
| Chennai | 78,301.80 | 195,508.80 | 93,817.45 | 27,675.40 | 68,325.70 | 463,629.15 |
| Delhi | 117,271.70 | 158,549.10 | 112,658.05 | 23,471.50 | 90,879.45 | 502,829.80 |
| Hyderabad | 83,731.00 | 228,913.10 | 109,475.40 | 24,800.05 | 81,670.35 | 528,589.90 |
| Mumbai | 75,209.85 | 214,738.40 | 78,970.15 | 38,018.00 | 72,122.10 | 479,058.50 |
| Pune | 90,349.55 | 187,629.25 | 108,456.60 | 34,275.15 | 80,207.50 | 500,918.05 |
| Total | 531,938.95 | 1,219,248.10 | 614,293.30 | 168,433.95 | 464,754.50 | 2,998,668.80 |

Pivot grand total INR 2,998,668.80 vs SQL Q2 total INR 2,998,668.80 -> equal.

**Per city (completed orders, named aggregation)**

| city | total | mean | count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1894 |
| Chennai | 463,629.15 | 272.72 | 1700 |
| Delhi | 502,829.80 | 243.62 | 2064 |
| Hyderabad | 528,589.90 | 284.95 | 1855 |
| Mumbai | 479,058.50 | 297.00 | 1613 |
| Pune | 500,918.05 | 252.48 | 1984 |

**Completed order_amount statistics**

| statistic | value |
|---|---|
| mean | 269.91 |
| median | 135.00 |
| skewness | 4.61 |
| Q1 | 64.80 |
| Q3 | 297.00 |
| lower fence | -283.50 |
| upper fence | 645.30 |
| outliers below lower fence | 0 |
| outliers above upper fence | 897 |

**Correlation matrix (completed orders)**

| index | order_amount | discount_pct | quantity | delivery_minutes |
|---|---|---|---|---|
| order_amount | 1.000 | -0.095 | 0.087 | -0.003 |
| discount_pct | -0.095 | 1.000 | -0.000 | -0.010 |
| quantity | 0.087 | -0.000 | 1.000 | -0.013 |
| delivery_minutes | -0.003 | -0.010 | -0.013 | 1.000 |

Largest absolute correlation: order_amount vs discount_pct = -0.095.

**Four questions (<=200 words)**

1. *What is the data about?* Quick-commerce orders in six Indian cities (Apr-Sep 2025): about 11,100 completed orders across groceries, dairy, snacks, personal care and electronics.
2. *Distributions and statistics?* Order amounts are strongly right-skewed: mean INR 269.91 versus median INR 135.00, skewness 4.61, middle half between INR 64.80 and INR 297.00.
3. *Outliers or anomalies?* No value lies below the lower fence (INR -283.50), but 897 orders exceed the upper fence of INR 645.30; these are mostly legitimate electronics purchases, not errors. Earlier anomalies (negative delivery times, text amounts, duplicates) were cleaned.
4. *Relationships?* All correlations are weak; the largest in absolute value is order_amount with discount_pct (-0.095); order_amount with quantity is +0.087. Correlation of this size does not establish causation.

## Part B

### B.1 RFM table (snapshot 2025-10-01 00:00:00)

| figure | value | reference |
|---|---|---|
| customers | 1,175 | 1,175 |
| recency median / min / max | 19 / 0 / 180 | 19 / 0 / 180 |
| frequency mean / max | 9.46 / 33 | 9.46 / 33 |
| monetary mean / max | 2,552.06 / 14,766.55 | 2,552.06 / 14,766.55 |

### B.2 K-means on standardised RFM (n_init=10, random_state=42)

| K | inertia | silhouette | reference (inertia / silhouette) | within tolerance |
|---|---|---|---|---|
| 2 | 1,820.8 | 0.465 | 1820.8 / 0.465 | yes |
| 3 | 1,268.2 | 0.445 | 1268.2 / 0.445 | yes |
| 4 | 743.1 | 0.526 | 743.1 / 0.526 | yes |
| 5 | 617.4 | 0.480 | 617.4 / 0.481 | yes |
| 6 | 503.2 | 0.482 | 503.2 / 0.482 | yes |
| 7 | 408.5 | 0.416 | 408.5 / 0.416 | yes |
| 8 | 366.5 | 0.412 | 366.5 / 0.412 | yes |

**Reading the elbow and choosing K.** Inertia drops steeply from K=2 to K=4 (1,820.8 -> 1,268.2 -> 743.1) and then flattens (743 -> 617 -> 503), so the elbow is at K=4. K=4 also has the highest silhouette score (0.526) of all K from 2 to 8, and four segments is a number a retention team can run campaigns for. **Final K = 4.**

**Why K=8 has lower inertia than K=2 yet that cannot pick K.** Inertia is the within-cluster sum of squares, and every extra centroid can only move points closer to some centroid, so inertia falls monotonically with K and reaches 0 when K equals the number of customers. A lower inertia therefore says nothing about whether the clusters are meaningful or actionable; we need the elbow, the silhouette score and business interpretability.

### B.3 Segment profile (K=4)

Cluster numbers are only labels produced by K-means; they carry no order or ranking, and can change with the seed or data.

| segment_id | segment_name | size | mean_recency | mean_frequency | mean_monetary |
|---|---|---|---|---|---|
| 0 | Occasional Low-Value | 436 | 28.86 | 3.77 | 1,057.62 |
| 1 | Premium High-Spenders | 138 | 24.12 | 6.77 | 7,129.70 |
| 2 | Loyal Frequent Regulars | 419 | 8.77 | 19.40 | 3,410.02 |
| 3 | Lapsing / At-Risk | 182 | 109.04 | 2.20 | 685.99 |

| segment_name | recommended_action |
|---|---|
| Occasional Low-Value | Low-cost nudges: category recommendations and first-repeat-order incentives. |
| Premium High-Spenders | VIP treatment: priority support, early access to electronics deals, no-discount loyalty perks. |
| Loyal Frequent Regulars | Protect and grow basket size: subscriptions / bundles for staples, free-delivery thresholds. |
| Lapsing / At-Risk | Win-back campaign: personalised reminder plus a time-limited coupon before they churn fully. |

Premium segment (highest mean monetary): segment_id 1, 138 customers, mean monetary INR 7,129.70, mean frequency 6.77.

Reference for K=4: sizes about 436 / 138 / 419 / 182; the 138-customer segment has mean monetary about 7,130 and mean frequency about 6.8; the 419-customer segment has mean frequency about 19.4 and mean recency about 8.8 days.

**PCA (2 components) on the scaled RFM matrix:** explained_variance_ratio_ = 0.634, 0.218; total 0.852 (reference about 0.634, 0.218, total 0.852).

Wrote `outputs/customer_segments.csv` (1,175 rows; `is_premium` is true for exactly segment 1).

### B.4 Lapse problem definition

| figure | value | reference |
|---|---|---|
| cohort size | 1,156 | 1,156 |
| lapsed | 443 (38.32%) | 443 (38.32%) |
| training rows | 924 | 924 |
| test rows | 232 | 232 |
| baseline accuracy (always 'not lapsed') | 61.6% | 61.6% |
| baseline lapsed-class F1 | 0.00 | 0 |

Cutoff 2025-09-01. Label `lapsed` = 1 if the customer has no completed order dated 2025-09-01 to 2025-09-30. Features: recency (days from cutoff to last completed order), frequency, monetary, average discount_pct, cancellation rate (cancelled / all final-status orders before the cutoff), average delivery_minutes, favourite category, city, age, gender, tenure_days (cutoff - signup_date). Every feature uses only orders dated before 2025-09-01; the segment labels from B.3 are **not** used.

### B.5 Preprocessing without leakage

Scaler mean of `frequency` (learned on the 924 training rows): 8.1147; full-cohort mean of `frequency`: 8.0285. They differ because the scaler only ever sees the training rows (the 232 test rows are held out), so its mean is the training-sample mean, not the whole-cohort mean.

Missing values in the feature matrix: 0 (asserted to be 0).

One-hot encoding uses `OneHotEncoder(drop="first")` (k-1 columns per variable) and both encoder and scaler live inside a scikit-learn `Pipeline`/`ColumnTransformer`, so they are fitted on training rows only (and, inside GridSearchCV, only on each fold's training part).

### B.6 Train, tune and evaluate

| model | grid | best params | best CV F1 |
|---|---|---|---|
| LogisticRegression(max_iter=1000) | C in {0.01, 0.1, 1, 10, 100} | {'C': 100} | 0.651 |
| RandomForestClassifier(n_estimators=100, random_state=42) | max_depth in {3, 5, 8} x min_samples_leaf in {5, 10, 20} (45 fits) | {'max_depth': 8, 'min_samples_leaf': 10} | 0.681 |

**Untouched test set, lapsed (positive) class** - baseline: precision 0.000, recall 0.000, F1 0.000

| model | confusion matrix [[TN, FP], [FN, TP]] | precision | recall | F1 |
|---|---|---|---|---|
| Logistic regression | [[111, 32], [31, 58]] | 0.644 | 0.652 | 0.648 |
| Random forest | [[113, 30], [31, 58]] | 0.659 | 0.652 | 0.655 |

**Selected model: Random forest** (test F1 0.655, recall 0.652, precision 0.659; required >= 0.55).

Justification: a missed lapsing customer is lost lifetime revenue, while a loyal customer wrongly targeted only costs one coupon or message, so the retention team should favour recall over precision. The selected model has the higher lapsed-class F1 on the untouched test set (same recall as the other model, slightly higher precision), and it comfortably beats the 0 F1 of the do-nothing baseline. It also captures non-linear effects (e.g. recency interacting with frequency) that a linear model cannot, which is why it wins here. The test set was used only once tuning was finished.

### B.7 Class-balance experiment (training set only)

Balanced training set: 570 not-lapsed and 570 lapsed rows (216 duplicated lapsed rows added; reference 570 each).

| model | precision | recall | F1 | confusion matrix |
|---|---|---|---|---|
| LR un-balanced | 0.644 | 0.652 | 0.648 | [[111, 32], [31, 58]] |
| LR over-sampled | 0.589 | 0.854 | 0.697 | [[90, 53], [13, 76]] |

The metric that moved most was **recall** (+0.202); precision changed by -0.055. Over-sampling makes the model flag more customers as lapsing: it catches many more true lapsers but also targets more loyal customers, so the retention team gains coverage at the price of a similar hit rate (precision does not improve) and a bigger campaign list. Reference: recall about 0.64 -> 0.85 with precision near 0.6.

### B.8 Interpretation, persistence and retention list

**Top five random-forest feature importances**

| feature | importance | business reading |
|---|---|---|
| num__frequency | 0.280 | habitual buyers keep ordering, so high frequency protects against lapsing |
| num__monetary | 0.271 | high lifetime spend tracks engaged, high-value customers |
| num__recency | 0.115 | the longer since the last order, the more likely the customer has already drifted away |
| num__avg_delivery_minutes | 0.061 | slow deliveries are a service-quality signal that can push people away |
| num__cancel_rate | 0.052 | customers whose orders get cancelled are more exposed to leaving |

**Logistic-regression coefficient signs** (standardised features; best LR)

| feature | coefficient | sign | business reading |
|---|---|---|---|
| num__recency | +0.040 | positive (raises lapse risk) | almost no effect once frequency is known (tiny coefficient); long silence matters mostly through low frequency |
| num__frequency | -1.698 | negative (lowers lapse risk) | the strongest signal: habitual buyers keep ordering, so more orders sharply lowers lapse risk |
| num__monetary | +0.285 | positive (raises lapse risk) | positive only after holding frequency fixed: big-ticket, low-frequency buyers (e.g. electronics) order rarely, so each month is more likely to be empty |

Saved `outputs/lapse_model.joblib` (Random forest pipeline incl. preprocessing). Reloaded model reproduces the original test predictions exactly: **True**.

Retention list: 40 test-set customers are in the Premium segment; the top 10 by predicted lapse probability were written to `outputs/retention_list.csv`; **1 of 10** truly lapsed.

**The Premium flag comes from the whole-window RFM of B.3, which includes September (the window that defines the lapse label), so it is a post-hoc whole-window label: this retention list is illustrative and not a leakage-free evaluation.** (The model's own features still use only orders before 2025-09-01.)

### B.9 Leakage statement

- **No encoder, scaler or resampler saw test rows.** The `StandardScaler` and `OneHotEncoder(drop="first")` live inside a scikit-learn `Pipeline` and are fitted on training rows only
  (inside `GridSearchCV` only on each fold's training part); the oversampler duplicated lapsed rows of the training set only, and the test set stayed unchanged.
- **No data from 2025-09-01 onwards entered the features:** every feature is computed from orders dated before the cutoff (asserted in code) plus the clean customer table.
- **The segment labels from B.3 were not used as features.**
- The Premium flag (B.3) is a post-hoc whole-window label, so the retention list is illustrative and not a leakage-free evaluation (see Limitations and Next Steps).

## Part C

### C.1 Input guardrails

Empty or whitespace-only tickets -> `skipped_empty`: 4. Tickets with est_tokens > 400 -> `flagged_long`: 4 (each estimates to [734] tokens). Neither kind is sent to the model.

### C.2 Redaction

| type | substitutions | reference |
|---|---|---|
| email | 55 | 55 |
| phone | 91 | 91 |
| order id | 132 | 132 |
| total | 278 | 278 |

Re-applying the three patterns to the redacted texts finds **0** matches (reference 0). Patterns are applied in the order email, phone, order id; the user message sent to the model is exactly the redacted ticket text.

### C.3 Prompts

Zero-shot system prompt:

```
You CLASSIFY customer support tickets. Allowed categories: billing, delivery, technical, fraud. Reply with JSON only, exactly of the form {"category": "...", "reason": "..."}.
```

Few-shot system prompt:

```
You CLASSIFY customer support tickets. Allowed categories: billing, delivery, technical, fraud. Reply with JSON only, exactly of the form {"category": "...", "reason": "..."}.
EXAMPLES:
Ticket: I was charged twice for [ORDER_ID]. Please refund the extra payment. -> {"category": "billing", "reason": "double charge"}
Ticket: Someone hacked my account and used my wallet balance. -> {"category": "fraud", "reason": "unauthorised account use"}
Ticket: The app freezes when I apply a coupon. -> {"category": "technical", "reason": "app fault"}
Ticket: My order has not arrived and the rider is not picking up. -> {"category": "delivery", "reason": "missing delivery"}
```

Prompt length: zero-shot 30 estimated tokens, few-shot 121 estimated tokens.

### C.4 Resilient calls

Each full run (after `mock_llm.reset()`) raised exactly 19 TransientLLMErrors (zero-shot) and 19 (few-shot) over the 156 eligible tickets; all were recovered by a retry (max 3 retries, wait = 0.05 x 2^n x uniform(0.9, 1.15) s) and no ticket ended with a permanent error. Every retry was retry number 0 and its wait lay in [0.0451, 0.0572] s (allowed 0.045 - 0.0575). One log line per retry is saved in `outputs/retry_log.txt`.

### C.5 Output validation (one full run over the 156 eligible tickets)

| status | zero-shot | few-shot | reference |
|---|---|---|---|
| classified_json | 135 | 135 | 135 |
| classified_keyword | 12 | 12 | 12 |
| flagged_invalid_output | 9 | 9 | 9 |

A reply that is valid JSON with a category outside the list (the mock's `shipping`) or a chatty reply with zero or several category words is `flagged_invalid_output` and goes to human review; the pipeline never guesses. Keyword extraction uses `re.findall(r"[a-z]+", text.lower())`, so `billing-related` yields the words `billing` and `related`.

### C.6 Zero-shot versus few-shot

| run | tickets classified | correct | accuracy % | total tokens | tokens per classified ticket |
|---|---|---|---|---|---|
| zero-shot | 147 | 126 | 85.7 | 9,088 | 61.8 |
| few-shot | 147 | 147 | 100.0 | 23,284 | 158.4 |

**Confusion table, zero-shot** (rows = true category, columns = predicted)

| true \ predicted | billing | delivery | technical | fraud |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| technical | 10 | 0 | 29 | 0 |
| fraud | 11 | 0 | 0 | 23 |

**When are few-shot tokens worth it?** The few-shot prompt costs 14,196 more tokens over the run (96.6 more per classified ticket, 156% more) but lifts accuracy from 85.7% to 100.0%: the zero-shot run misclassified 21 tickets (all fraud or technical tickets predicted as billing), which are exactly the cases where a mistake is expensive (a fraud report not reaching the security team). For a high-stakes category like fraud the extra ~97 tokens per ticket are clearly worth paying; for a low-stakes triage where billing/technical confusion is cheap to fix, or at very high volume with a tight budget, the cheaper zero-shot prompt may be enough. With the mock model the gain comes from the order in which keyword groups are checked.

### C.7 Token budget and routing

Zero-shot run with TOKEN_BUDGET = 6000: the running total reached the budget after 103 model calls (6,018 tokens) and **53 tickets ended `budget_exhausted`** (no call made, routed to human review). Saved as `outputs/triage_budget_demo.csv`.

`outputs/triage_results.csv` comes from a few-shot run with TOKEN_BUDGET = 100000 (no ticket exhausted: 0). Routing rules: skipped_empty / flagged_long / flagged_invalid_output / budget_exhausted -> `human_review` (NORMAL); else fraud -> `security_team` (HIGH); else Premium customer -> `premium_desk` (HIGH); else `support_queue` (NORMAL). A customer with no row in the segment file counts as not premium.

| route | priority | tickets |
|---|---|---|
| human_review | NORMAL | 17 |
| premium_desk | HIGH | 11 |
| security_team | HIGH | 34 |
| support_queue | NORMAL | 102 |

| status | tickets |
|---|---|
| classified_json | 135 |
| classified_keyword | 12 |
| flagged_invalid_output | 9 |
| skipped_empty | 4 |
| flagged_long | 4 |

**Rule-validated vs human:** 147 tickets were classified and passed output validation (routed automatically); 17 went to human review (4 empty, 4 too long, 9 invalid model output).

Privacy assertion: none of the 571 user messages sent to the model across all four runs contains an email, a phone number or an order id (asserted in code); no empty or long ticket triggered a model call (model calls per run = 156 eligible tickets).

## Part D

### D.1 Chunking (40 words, overlap 10 words)

| document | chunks | reference |
|---|---|---|
| refund_policy.txt | 4 | 4 |
| delivery_policy.txt | 3 | 3 |
| escalation_policy.txt | 3 | 3 |
| payment_policy.txt | 3 | 3 |
| privacy_policy.txt | 2 | 2 |

Total chunks: 15 (reference 15). Every chunk records its source file name. With overlap 0 there are 13 chunks.

### D.2 TF-IDF retrieval

Query `refund policy late delivery` -> top three: refund_policy.txt (0.272), refund_policy.txt (0.251), refund_policy.txt (0.245) (all three from refund_policy.txt, as required).

**TF and IDF.** Term frequency (TF) measures how often a word occurs inside one chunk, while inverse document frequency (IDF) measures how rare the word is across all chunks, so words that are frequent in a chunk but rare overall get the highest weights. A limitation is that TF-IDF matches exact words only: two phrasings with the same meaning but different words (for example "money back" versus "refund") get no similarity, and a question whose key words are split across a chunk boundary can retrieve the wrong chunk.

### D.3 Augment and answer

`answer(question)` retrieves three chunks, then sends two messages to `mock_llm.chat`: a system message containing the words POLICY ASSISTANT (answer only from the context, otherwise say the information could not be found) and a user message `CONTEXT:` + newline + the three chunks each prefixed with `[<source file>]` on its own line + newline `QUESTION:` + newline + the question. It returns the model text and the sources used.

Example: *How many days do I have to raise a refund request?* -> "A refund request must be raised within 7 days of the delivery date." (sources: refund_policy.txt, privacy_policy.txt, refund_policy.txt).

### D.4 Retrieval and grounding evaluation (14 questions)

| measure | overlap 10 | overlap 0 | reference (overlap 10) |
|---|---|---|---|
| answerable: expected_doc in top-3 sources | 12/12 | 12/12 | 12/12 |
| answerable: answer contains expected_fact | 12/12 | 10/12 | 12/12 |
| unanswerable: fallback 'I could not find...' | 2/2 | 2/2 | 2/2 |

With zero overlap 2 answerable question(s) break:

- *What delivery time makes a completed order eligible for a full refund?* (expected fact "more than 45 minutes" in refund_policy.txt): the answer was "receives a full refund of the order amount.". The sentence that holds the fact is cut by a chunk boundary: no single 40-word chunk contains the whole sentence ("If a completed order is delivered in more than 45 minutes, the customer receives a full refund of the order amount."), so the model only sees a fragment of it and answers with the wrong half.
- *How soon is money returned when a payment fails but is deducted?* (expected fact "48 hours" in payment_policy.txt): the answer was "If a payment fails but money is deducted, the amount is returned automatically 1000 must be approved by a human support lead before money is returned.". The sentence that holds the fact is cut by a chunk boundary: no single 40-word chunk contains the whole sentence ("If a payment fails but money is deducted, the amount is returned automatically within 48 hours."), so the model only sees a fragment of it and answers with the wrong half.

Overlap keeps the words around a chunk boundary in two neighbouring chunks, so an answer that straddles the boundary still appears whole in at least one chunk.

### D.5 Tools

| tool | description | parameters |
|---|---|---|
| get_order | Look up an order's status, amount and delivery time in the Instamart database. | order_id |
| search_policy | Search the company policy documents and return the three best matching chunks. | query |
| calculate_refund | Compute the refund in INR from the order amount and the delivery time following the refund policy. | order_amount, delivery_minutes |
| issue_refund | Issue an automatic refund of up to INR 1000 and record it in the ledger. | order_id, refund_inr |
| escalate_to_human | Escalate a refund above INR 1000 to a human support lead and record it in the ledger. | order_id, refund_inr |

`TOOL_MAP` maps each tool name to its Python function; the loop looks up the name the model asks for and calls the function with the arguments dictionary. Refund rules implemented in the tools: only completed orders are refunded (the model path denies other statuses), delivery > 45 min -> 100% of order_amount, otherwise 0, and refunds above INR 1000 go through `escalate_to_human` instead of `issue_refund`. `issue_refund` / `escalate_to_human` append one JSON line (timestamp, tool, run_id, order_id, refund_inr) to `outputs/refund_ledger.jsonl`.

### D.6 ReAct loop - the seven scenarios (also in `outputs/agent_runs.jsonl`)

| scenario id | status | refund INR | expected refund INR | model calls | tokens used | as expected |
|---|---|---|---|---|---|---|
| S1 | REFUND | 128.25 | 128.25 | 5 | 1057 | yes |
| S2 | REFUND | 135.00 | 135.00 | 5 | 1057 | yes |
| S3 | ESCALATE | 1,999.00 | 1,999.00 | 5 | 1044 | yes |
| S4 | DENIED | 0.00 | 0.00 | 4 | 733 | yes |
| S5 | DENIED | 0.00 | 0.00 | 3 | 433 | yes |
| S6 | NOT_FOUND | 0.00 | 0.00 | 2 | 136 | yes |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 0.00 | 6 | 3192 | yes |

S7 (`[LOOP]`) keeps asking for `search_policy`, finishes all six model calls (3,192 estimated tokens, below the default limit of 6000) and ends `FALLBACK_MAX_ITERATIONS` with a human-handoff summary.

### D.7 Stop safely

| run | status | model calls | tokens used | tools executed |
|---|---|---|---|---|
| S7 default (max_iterations=6, token_limit=6000) | FALLBACK_MAX_ITERATIONS | 6 | 3192 | 6 |
| S1 with token_limit=200 | FALLBACK_TOKEN_LIMIT | 3 | 450 | 2 |
| S1 with time_limit_seconds=0 | FALLBACK_TIMEOUT | 0 | 0 | 0 |

- Token-limit demo: the limit was exceeded after model call 3 (450 tokens > 200), the tool that call requested was **not** executed. final_answer: "HANDOFF to human agent: automated run stopped (token limit 200 exceeded). Request: My order ORD-100048 reached me very late. I would like a refund. Steps completed: get_order -> search_policy. Model calls: 3, tokens: 450."
- Timeout demo: `elapsed >= time_limit_seconds` is true at the top of the first iteration (0 >= 0), so the run stops before any model call. final_answer: "HANDOFF to human agent: automated run stopped (time limit of 0 s reached). Request: My order ORD-100048 reached me very late. I would like a refund. Steps completed: no tools. Model calls: 0, tokens: 0."

### D.8 Resumable run (crash after the second tool, then resume)

Simulated crash: *simulated crash after tool #2 (search_policy)*. Checkpoint `state/S1_crash_resume_demo.json` holds run_id, scenario_id, the full message list (6 messages), tokens_used (166), step (2) and tools already executed (get_order, search_policy).

`resume_agent('S1_crash_resume_demo')` loaded the checkpoint and continued: final status **REFUND**, refund INR 128.25, 5 model calls in total.

**Tools executed across the crash and the resume, in order (4 executions, none repeated):** `get_order` -> `search_policy` -> `calculate_refund` -> `issue_refund`.

### D.9 ReAct versus plan-and-execute

Measured with the mock model, my ReAct agent needs 5 model calls for S1 (1,057 estimated tokens), 4 for S4, 3 for S5 (433 tokens) and 2 for S6 (136 tokens). Every call re-sends the whole growing message history, so tokens rise faster than calls. A plan-and-execute design asks the model once for the whole step list (get_order, search_policy, calculate_refund, issue_refund) plus at most one call to write the answer, so it needs about 2 calls and clearly fewer tokens, and the model is not re-invoked after every tool.

Its weakness is the unexpected observation. In S6 (NOT_FOUND) and S5 (a cancelled order) the plan made before seeing the data is wrong: the remaining steps must be skipped or the plan revised, and a blindly executed tail could calculate or even issue a refund for an order that is not refundable. The ReAct agent reacts immediately: it stops after 2 calls for NOT_FOUND and after 3 calls for a cancelled order, and routes a refund above INR 1000 to a human only after seeing the computed amount.

For this refund workflow I would choose ReAct (with the hard limits of D.7): the chains are short (at most 5 calls), money-moving steps depend on earlier observations, and the extra tokens are cheap compared with one wrong refund. Plan-and-execute becomes attractive only for long, predictable workflows.

_(word count: 224)_

Outputs: `outputs/agent_runs.jsonl` (7 lines, one per scenario run), `outputs/agent_demos.jsonl` (the three demonstration runs), `outputs/refund_ledger.jsonl` (4 issued/escalated actions, each tagged with its run_id; the crash-and-resume demo adds one `issue_refund` for S1) and `state/` (9 checkpoint files).

## Limitations and Next Steps

- **Synthetic data and a mock model.** All results come from generated data and the deterministic `mock_llm.py`. The accuracy of the classifier (85.7% / 100%) and the perfect agent behaviour
  reflect the mock's keyword rules, not a real hosted model, and real tickets would be noisier.
- **The Premium flag is a post-hoc whole-window label.** It comes from the RFM of B.3, which includes September, the very window that defines the lapse label, so the retention list
  (`outputs/retention_list.csv`) is illustrative and not a leakage-free evaluation. A proper evaluation would compute the segments from orders before the cutoff only.
- **Imputation choices.** Missing ages are filled with the city median and missing delivery times with the product-category median among completed orders; this reduces variance and can bias
  downstream statistics. Cancelled orders keep a missing delivery time on purpose.
- **Small test set.** The lapse test set has 232 customers, so F1 differences of a few points between models are within noise; cross-validated scores are more trustworthy than one split.
  Over-sampling raised recall at the cost of precision, which suits a cheap campaign but not an expensive one.
- **TF-IDF retrieval** matches exact words only and depends on chunk boundaries (shown by the zero-overlap run); embeddings or hybrid search would handle paraphrases better.
- **Agent safety.** Refund rules are enforced in tools, but the 7-day window is assumed satisfied and `issue_refund` only writes a ledger line; a real system needs idempotency keys,
  authentication and an audit trail.
- **Next steps:** time-based validation for the lapse model, probability calibration and cost-based thresholding, evaluating the triage on noisy real tickets, and swapping the mock for a hosted model behind the same guardrails.
