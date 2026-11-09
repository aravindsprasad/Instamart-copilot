"""Part B tasks 4-8: lapse definition, leak-free preprocessing, tuning, evaluation, class-balance experiment, interpretation."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample
from util import Report, md_table

rep = Report("part_b_lapse")
CUTOFF, SEPT_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-10-01")

orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
cust = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
prod = pd.read_csv("data/clean/products.csv")
orders = orders.merge(prod[["product_id", "category"]], on="product_id")

# ------------------------------------------------------------------ task 4: cohort, label, features
rep.add("### B.4 Lapse problem definition")
before = orders[orders.order_ts < CUTOFF]                       # ONLY orders before the cutoff feed the features
done_b = before[before.status == "completed"]
cohort_ids = done_b.customer_id.unique()
sept = orders[(orders.status == "completed") & (orders.order_ts >= CUTOFF) & (orders.order_ts < SEPT_END)]
y = pd.Series(~pd.Index(cohort_ids).isin(sept.customer_id.unique()), index=pd.Index(cohort_ids), name="lapsed").astype(int)

g = done_b.groupby("customer_id")
feat = pd.DataFrame({
    "recency": (CUTOFF - g["order_ts"].max()).dt.days,
    "frequency": g["order_id"].nunique(),
    "monetary": g["order_amount"].sum(),
    "avg_discount_pct": g["discount_pct"].mean(),
    "avg_delivery_minutes": g["delivery_minutes"].mean(),
    "fav_category": g["category"].agg(lambda s: s.value_counts().sort_index().sort_values(ascending=False, kind="stable").index[0]),
})
final_b = before[before.status.isin(["completed", "cancelled"])]
feat["cancel_rate"] = final_b.groupby("customer_id")["status"].apply(lambda s: (s == "cancelled").mean()).reindex(feat.index).fillna(0)
c = cust.set_index("customer_id")
feat["city"], feat["age"], feat["gender"] = c.loc[feat.index, "city"], c.loc[feat.index, "age"], c.loc[feat.index, "gender"]
feat["tenure_days"] = (CUTOFF - c.loc[feat.index, "signup_date"]).dt.days
X = feat.loc[y.index]
NUM = ["recency", "frequency", "monetary", "avg_discount_pct", "cancel_rate", "avg_delivery_minutes", "age", "tenure_days"]
CAT = ["city", "gender", "fav_category"]
X = X[NUM + CAT]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
base_pred = np.zeros(len(yte), dtype=int)
base_acc = accuracy_score(yte, base_pred)
base_f1 = f1_score(yte, base_pred, zero_division=0)
rep.add(md_table(pd.DataFrame({
    "figure": ["cohort size", "lapsed", "training rows", "test rows", "baseline accuracy (always 'not lapsed')", "baseline lapsed-class F1"],
    "value": [f"{len(y):,}", f"{int(y.sum())} ({y.mean() * 100:.2f}%)", f"{len(Xtr)}", f"{len(Xte)}", f"{base_acc * 100:.1f}%", f"{base_f1:.2f}"],
    "reference": ["1,156", "443 (38.32%)", "924", "232", "61.6%", "0"]})))
rep.add("Cutoff 2025-09-01. Label `lapsed` = 1 if the customer has no completed order dated 2025-09-01 to 2025-09-30. Features: "
        "recency (days from cutoff to last completed order), frequency, monetary, average discount_pct, cancellation rate "
        "(cancelled / all final-status orders before the cutoff), average delivery_minutes, favourite category, city, age, gender, "
        "tenure_days (cutoff - signup_date). Every feature uses only orders dated before 2025-09-01; the segment labels from B.3 are **not** used.")
assert (before.order_ts < CUTOFF).all() and not any(c_ in X.columns for c_ in ["segment_id", "segment_name", "is_premium"])

# ------------------------------------------------------------------ task 5: leak-free preprocessing
rep.add("### B.5 Preprocessing without leakage")
def make_pre():
    return ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT)])
def make_pipe(clf):
    return Pipeline([("pre", make_pre()), ("clf", clf)])

probe = make_pre().fit(Xtr)                                      # fitted on TRAIN rows only
sc_mean = probe.named_transformers_["num"].mean_[NUM.index("frequency")]
rep.add(f"Scaler mean of `frequency` (learned on the {len(Xtr)} training rows): {sc_mean:.4f}; full-cohort mean of `frequency`: "
        f"{X['frequency'].mean():.4f}. They differ because the scaler only ever sees the training rows (the 232 test rows are held out), "
        "so its mean is the training-sample mean, not the whole-cohort mean.")
rep.add(f"Missing values in the feature matrix: {int(X.isna().sum().sum())} (asserted to be 0).")
assert X.isna().sum().sum() == 0
rep.add("One-hot encoding uses `OneHotEncoder(drop=\"first\")` (k-1 columns per variable) and both encoder and scaler live inside a scikit-learn "
        "`Pipeline`/`ColumnTransformer`, so they are fitted on training rows only (and, inside GridSearchCV, only on each fold's training part).")

# ------------------------------------------------------------------ task 6: tune & evaluate
rep.add("### B.6 Train, tune and evaluate")
def metrics(model, Xe, ye):
    p = model.predict(Xe)
    return dict(cm=confusion_matrix(ye, p), prec=precision_score(ye, p, zero_division=0),
                rec=recall_score(ye, p, zero_division=0), f1=f1_score(ye, p, zero_division=0))

lr_gs = GridSearchCV(make_pipe(LogisticRegression(max_iter=1000)), {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(Xtr, ytr)
rf_gs = GridSearchCV(make_pipe(RandomForestClassifier(n_estimators=100, random_state=42)),
                     {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]}, cv=5, scoring="f1").fit(Xtr, ytr)
rep.add(md_table(pd.DataFrame({
    "model": ["LogisticRegression(max_iter=1000)", "RandomForestClassifier(n_estimators=100, random_state=42)"],
    "grid": ["C in {0.01, 0.1, 1, 10, 100}", "max_depth in {3, 5, 8} x min_samples_leaf in {5, 10, 20} (45 fits)"],
    "best params": [str({k.split('__')[1]: v for k, v in lr_gs.best_params_.items()}), str({k.split('__')[1]: v for k, v in rf_gs.best_params_.items()})],
    "best CV F1": [f"{lr_gs.best_score_:.3f}", f"{rf_gs.best_score_:.3f}"]})))
m_lr, m_rf = metrics(lr_gs.best_estimator_, Xte, yte), metrics(rf_gs.best_estimator_, Xte, yte)
rep.add("**Untouched test set, lapsed (positive) class** - baseline: precision 0.000, recall 0.000, F1 0.000")
rep.add(md_table(pd.DataFrame({
    "model": ["Logistic regression", "Random forest"],
    "confusion matrix [[TN, FP], [FN, TP]]": [str(m_lr["cm"].tolist()), str(m_rf["cm"].tolist())],
    "precision": [f"{m_lr['prec']:.3f}", f"{m_rf['prec']:.3f}"], "recall": [f"{m_lr['rec']:.3f}", f"{m_rf['rec']:.3f}"],
    "F1": [f"{m_lr['f1']:.3f}", f"{m_rf['f1']:.3f}"]})))
chosen_name, chosen, m_ch = ("Random forest", rf_gs.best_estimator_, m_rf) if m_rf["f1"] >= m_lr["f1"] else ("Logistic regression", lr_gs.best_estimator_, m_lr)
rep.add(f"**Selected model: {chosen_name}** (test F1 {m_ch['f1']:.3f}, recall {m_ch['rec']:.3f}, precision {m_ch['prec']:.3f}; required >= 0.55).\n\n"
        f"Justification: a missed lapsing customer is lost lifetime revenue, while a loyal customer wrongly targeted only costs one coupon or message, "
        f"so the retention team should favour recall over precision. The selected model has the higher lapsed-class F1 on the untouched test set (same recall as the other model, slightly higher precision), and it comfortably beats the 0 F1 of the do-nothing baseline. "
        "It also captures non-linear effects (e.g. recency interacting with frequency) that a linear model cannot, which is why it wins here. "
        "The test set was used only once tuning was finished.")
assert m_ch["f1"] >= 0.55

# ------------------------------------------------------------------ task 7: oversampling
rep.add("### B.7 Class-balance experiment (training set only)")
lr_best_params = {k.split("__")[1]: v for k, v in lr_gs.best_params_.items()}
n_add = int((ytr == 0).sum() - (ytr == 1).sum())
lap_idx = ytr.index[ytr == 1]
extra = resample(lap_idx, replace=True, n_samples=n_add, random_state=42)
bal_idx = list(ytr.index) + list(extra)
Xbal, ybal = Xtr.loc[bal_idx], ytr.loc[bal_idx]
lr_bal = make_pipe(LogisticRegression(max_iter=1000, **lr_best_params)).fit(Xbal, ybal)
m_bal = metrics(lr_bal, Xte, yte)
rep.add(f"Balanced training set: {int((ybal == 0).sum())} not-lapsed and {int((ybal == 1).sum())} lapsed rows ({n_add} duplicated lapsed rows added; reference 570 each).")
rep.add(md_table(pd.DataFrame({
    "model": ["LR un-balanced", "LR over-sampled"],
    "precision": [f"{m_lr['prec']:.3f}", f"{m_bal['prec']:.3f}"], "recall": [f"{m_lr['rec']:.3f}", f"{m_bal['rec']:.3f}"],
    "F1": [f"{m_lr['f1']:.3f}", f"{m_bal['f1']:.3f}"], "confusion matrix": [str(m_lr["cm"].tolist()), str(m_bal["cm"].tolist())]})))
d = {"precision": m_bal["prec"] - m_lr["prec"], "recall": m_bal["rec"] - m_lr["rec"], "F1": m_bal["f1"] - m_lr["f1"]}
mv = max(d, key=lambda k: abs(d[k]))
rep.add(f"The metric that moved most was **{mv}** ({d[mv]:+.3f}); precision changed by {d['precision']:+.3f}. Over-sampling makes the model flag more customers "
        "as lapsing: it catches many more true lapsers but also targets more loyal customers, so the retention team gains coverage at the price of a "
        "similar hit rate (precision does not improve) and a bigger campaign list. Reference: recall about 0.64 -> 0.85 with precision near 0.6.")

# ------------------------------------------------------------------ task 8: interpret, persist, use
rep.add("### B.8 Interpretation, persistence and retention list")
names = chosen.named_steps["pre"].get_feature_names_out()
if chosen_name == "Random forest":
    imp = pd.Series(chosen.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False).head(5)
else:
    imp = pd.Series(np.abs(chosen.named_steps["clf"].coef_[0]), index=names).sort_values(ascending=False).head(5)
rf_names = rf_gs.best_estimator_.named_steps["pre"].get_feature_names_out()
imp_rf = pd.Series(rf_gs.best_estimator_.named_steps["clf"].feature_importances_, index=rf_names).sort_values(ascending=False).head(5)
reading = {"num__recency": "the longer since the last order, the more likely the customer has already drifted away",
           "num__frequency": "habitual buyers keep ordering, so high frequency protects against lapsing",
           "num__monetary": "high lifetime spend tracks engaged, high-value customers",
           "num__tenure_days": "tenure carries a little information about how settled the customer is",
           "num__avg_delivery_minutes": "slow deliveries are a service-quality signal that can push people away",
           "num__avg_discount_pct": "discount sensitivity says how deal-driven the customer is",
           "num__age": "age has only a weak relation to lapse here (synthetic data)",
           "num__cancel_rate": "customers whose orders get cancelled are more exposed to leaving"}
rep.add("**Top five random-forest feature importances**")
rep.add(md_table(pd.DataFrame({"feature": imp_rf.index, "importance": [f"{v:.3f}" for v in imp_rf.values],
                               "business reading": [reading.get(f, "a customer-profile effect (city / gender / category dummy)") for f in imp_rf.index]})))
lr_names = lr_gs.best_estimator_.named_steps["pre"].get_feature_names_out()
coef = pd.Series(lr_gs.best_estimator_.named_steps["clf"].coef_[0], index=lr_names)
lr_reading = {"num__recency": "almost no effect once frequency is known (tiny coefficient); long silence matters mostly through low frequency",
              "num__frequency": "the strongest signal: habitual buyers keep ordering, so more orders sharply lowers lapse risk",
              "num__monetary": "positive only after holding frequency fixed: big-ticket, low-frequency buyers (e.g. electronics) order rarely, so each month is more likely to be empty"}
rows = []
for f in ["num__recency", "num__frequency", "num__monetary"]:
    sign = "positive (raises lapse risk)" if coef[f] > 0 else "negative (lowers lapse risk)"
    rows.append((f, f"{coef[f]:+.3f}", sign, lr_reading[f]))
rep.add("**Logistic-regression coefficient signs** (standardised features; best LR)")
rep.add(md_table(pd.DataFrame(rows, columns=["feature", "coefficient", "sign", "business reading"])))

joblib.dump(chosen, "outputs/lapse_model.joblib")
re = joblib.load("outputs/lapse_model.joblib")
same = bool(np.array_equal(chosen.predict(Xte), re.predict(Xte)) and np.allclose(chosen.predict_proba(Xte), re.predict_proba(Xte)))
rep.add(f"Saved `outputs/lapse_model.joblib` ({chosen_name} pipeline incl. preprocessing). Reloaded model reproduces the original test predictions exactly: **{same}**.")
assert same

seg = pd.read_csv("outputs/customer_segments.csv")
prem_ids = set(seg.loc[seg.is_premium, "customer_id"])
proba = pd.Series(chosen.predict_proba(Xte)[:, 1], index=Xte.index)
sel = proba[proba.index.isin(prem_ids)].sort_values(ascending=False).head(10)
ret = pd.DataFrame({"customer_id": sel.index, "lapse_probability": sel.values.round(4), "actually_lapsed": yte.loc[sel.index].values})
ret.to_csv("outputs/retention_list.csv", index=False)
n_prem_test = int(proba.index.isin(prem_ids).sum())
rep.add(f"Retention list: {n_prem_test} test-set customers are in the Premium segment; the top {len(ret)} by predicted lapse probability were written to "
        f"`outputs/retention_list.csv`; **{int(ret.actually_lapsed.sum())} of {len(ret)}** truly lapsed.")
rep.add("**The Premium flag comes from the whole-window RFM of B.3, which includes September (the window that defines the lapse label), so it is a "
        "post-hoc whole-window label: this retention list is illustrative and not a leakage-free evaluation.** (The model's own features still use only orders before 2025-09-01.)")
rep.save()
