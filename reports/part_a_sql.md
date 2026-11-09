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
