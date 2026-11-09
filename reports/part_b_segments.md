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
