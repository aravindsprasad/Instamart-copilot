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
