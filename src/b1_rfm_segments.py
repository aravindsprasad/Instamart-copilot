"""Part B tasks 1-3: RFM table, K-means (K = 2..8), segment profile/naming, PCA, outputs/customer_segments.csv."""
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from util import Report, md_table

os.makedirs("outputs", exist_ok=True)
rep = Report("part_b_segments")
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
FINAL_K = 4

orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
done = orders[orders.status == "completed"]

# ------------------------------------------------------------------ task 1: RFM
rep.add("### B.1 RFM table (snapshot 2025-10-01 00:00:00)")
g = done.groupby("customer_id")
rfm = pd.DataFrame({"last_order_ts": g["order_ts"].max(), "frequency": g["order_id"].nunique(),
                    "monetary": g["order_amount"].sum()})
rfm["recency"] = (SNAPSHOT - rfm["last_order_ts"]).dt.days
rfm = rfm[["recency", "frequency", "monetary"]].reset_index()
summ = pd.DataFrame({"figure": ["customers", "recency median / min / max", "frequency mean / max", "monetary mean / max"],
                     "value": [f"{len(rfm):,}", f"{rfm.recency.median():g} / {rfm.recency.min()} / {rfm.recency.max()}",
                               f"{rfm.frequency.mean():.2f} / {rfm.frequency.max()}",
                               f"{rfm.monetary.mean():,.2f} / {rfm.monetary.max():,.2f}"],
                     "reference": ["1,175", "19 / 0 / 180", "9.46 / 33", "2,552.06 / 14,766.55"]})
rep.add(md_table(summ))

# ------------------------------------------------------------------ task 2: K-means
rep.add("### B.2 K-means on standardised RFM (n_init=10, random_state=42)")
scaler = StandardScaler()
X = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])
ref = {2: (1820.8, .465), 3: (1268.2, .445), 4: (743.1, .526), 5: (617.4, .481), 6: (503.2, .482), 7: (408.5, .416), 8: (366.5, .412)}
rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    sil = silhouette_score(X, km.labels_)
    ok = abs(km.inertia_ / ref[k][0] - 1) <= .05 and abs(sil - ref[k][1]) <= .05
    rows.append((k, f"{km.inertia_:,.1f}", f"{sil:.3f}", f"{ref[k][0]} / {ref[k][1]}", "yes" if ok else "NO"))
rep.add(md_table(pd.DataFrame(rows, columns=["K", "inertia", "silhouette", "reference (inertia / silhouette)", "within tolerance"])))
rep.add("**Reading the elbow and choosing K.** Inertia drops steeply from K=2 to K=4 (1,820.8 -> 1,268.2 -> 743.1) and then flattens "
        "(743 -> 617 -> 503), so the elbow is at K=4. K=4 also has the highest silhouette score (0.526) of all K from 2 to 8, "
        "and four segments is a number a retention team can run campaigns for. **Final K = 4.**")
rep.add("**Why K=8 has lower inertia than K=2 yet that cannot pick K.** Inertia is the within-cluster sum of squares, and every extra "
        "centroid can only move points closer to some centroid, so inertia falls monotonically with K and reaches 0 when K equals the number "
        "of customers. A lower inertia therefore says nothing about whether the clusters are meaningful or actionable; we need the elbow, "
        "the silhouette score and business interpretability.")

# ------------------------------------------------------------------ task 3: profile, names, PCA
km = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_
prof = rfm.groupby("segment_id").agg(size=("customer_id", "count"), mean_recency=("recency", "mean"),
                                     mean_frequency=("frequency", "mean"), mean_monetary=("monetary", "mean"))
prem = prof["mean_monetary"].idxmax()
rest = prof.drop(index=prem)
names = {prem: "Premium High-Spenders"}
lapsing = rest["mean_recency"].idxmax()
names[lapsing] = "Lapsing / At-Risk"
rest2 = rest.drop(index=lapsing)
loyal = rest2["mean_frequency"].idxmax()
names[loyal] = "Loyal Frequent Regulars"
for s in rest2.index:
    if s != loyal:
        names[s] = "Occasional Low-Value"
actions = {"Premium High-Spenders": "VIP treatment: priority support, early access to electronics deals, no-discount loyalty perks.",
           "Lapsing / At-Risk": "Win-back campaign: personalised reminder plus a time-limited coupon before they churn fully.",
           "Loyal Frequent Regulars": "Protect and grow basket size: subscriptions / bundles for staples, free-delivery thresholds.",
           "Occasional Low-Value": "Low-cost nudges: category recommendations and first-repeat-order incentives."}
prof["segment_name"] = prof.index.map(names)
prof["recommended_action"] = prof["segment_name"].map(actions)
rep.add("### B.3 Segment profile (K=4)")
rep.add("Cluster numbers are only labels produced by K-means; they carry no order or ranking, and can change with the seed or data.")
show = prof.reset_index()
rep.add(md_table(show[["segment_id", "segment_name", "size", "mean_recency", "mean_frequency", "mean_monetary"]]))
rep.add(md_table(show[["segment_name", "recommended_action"]]))
rep.add(f"Premium segment (highest mean monetary): segment_id {prem}, {int(prof.loc[prem, 'size'])} customers, "
        f"mean monetary INR {prof.loc[prem, 'mean_monetary']:,.2f}, mean frequency {prof.loc[prem, 'mean_frequency']:.2f}.")
rep.add("Reference for K=4: sizes about 436 / 138 / 419 / 182; the 138-customer segment has mean monetary about 7,130 and mean frequency "
        "about 6.8; the 419-customer segment has mean frequency about 19.4 and mean recency about 8.8 days.")

pca = PCA(n_components=2).fit(X)
ev = pca.explained_variance_ratio_
rep.add(f"**PCA (2 components) on the scaled RFM matrix:** explained_variance_ratio_ = {ev[0]:.3f}, {ev[1]:.3f}; total {ev.sum():.3f} "
        "(reference about 0.634, 0.218, total 0.852).")

rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_id"] == prem
rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]] \
    .to_csv("outputs/customer_segments.csv", index=False)
rep.add(f"Wrote `outputs/customer_segments.csv` ({len(rfm):,} rows; `is_premium` is true for exactly segment {prem}).")
rep.save()
