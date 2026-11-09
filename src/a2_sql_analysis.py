"""Part A SQL analysis (Q1-Q7) and pandas cross-check / EDA."""
import re, sqlite3
import numpy as np
import pandas as pd
from util import Report, md_table

rep = Report("part_a_sql")
con = sqlite3.connect("data/instamart.db")
text = open("sql/queries.sql", encoding="utf-8").read()
queries = {m.group(1): m.group(2).strip() for m in
           re.finditer(r"^-- (Q\d):[^\n]*\n(.*?)(?=^-- Q\d:|\Z)", text, re.S | re.M)}
assert sorted(queries) == [f"Q{i}" for i in range(1, 8)]
R = {k: pd.read_sql_query(q, con) for k, q in queries.items()}

rep.add("### A.5 SQL analysis (queries in `sql/queries.sql`)")
rows = []
q1, q2, q3, q4, q5, q6, q7 = (R[f"Q{i}"] for i in range(1, 8))
rows.append(("Q1 WHERE then HAVING", f"{len(q1)} customers", "30"))
rows.append(("Q2 three-table join", f"{len(q2)} rows; sum INR {q2.revenue.sum():,.2f}; largest {q2.iloc[0].city} / {q2.iloc[0].category} INR {q2.iloc[0].revenue:,.2f}",
             "30 rows; INR 2,998,668.80; Bengaluru / Electronics INR 233,909.45"))
rows.append(("Q3 scalar subquery", f"{len(q3)} products", "11"))
rows.append(("Q4 LEFT JOIN", f"{len(q4)} customers without a completed order", "25"))
rows.append(("Q5 CTE", f"{len(q5)} customers above average INR {q5.avg_total.iloc[0]:,.2f}", "550; INR 2,552.06"))
top = q6[q6.row_num == 1].set_index("city")
n_diff = int((q6.rnk != q6.row_num).sum())
rows.append(("Q6 window ranking", f"{len(q6)} rows; {n_diff} rows with RANK != ROW_NUMBER", "8"))
rows.append(("Q7 running total", f"{len(q7)} days; last cumulative INR {q7.cumulative_revenue.iloc[-1]:,.2f}", "183; INR 2,998,668.80"))
rep.add(md_table(pd.DataFrame(rows, columns=["query", "result", "reference"])))
rep.add("**Q6 top customer per city:** " + "; ".join(f"{c} {r.customer_id} ({r.total_spend:,.2f})" for c, r in top.iterrows()))
rep.add("Why RANK() and ROW_NUMBER() differ: when two customers in a city have exactly the same total spend, RANK() gives them the "
        "same rank (and skips the next number) while ROW_NUMBER() still hands out a unique, arbitrary sequence number, so from the "
        "second tied row onwards the two columns disagree.")
assert len(q1) == 30 and len(q2) == 30 and len(q3) == 11 and len(q4) == 25 and len(q5) == 550 and n_diff == 8 and len(q7) == 183
assert abs(q2.revenue.sum() - 2998668.80) < 0.01 and abs(q7.cumulative_revenue.iloc[-1] - 2998668.80) < 0.01

# ------------------------------------------------------------------ pandas cross-check
rep.add("### A.6 pandas cross-check and exploratory analysis")
orders = pd.read_sql_query("select * from orders", con)
custs = pd.read_sql_query("select * from customers", con)
prods = pd.read_sql_query("select * from products", con)
done = orders[orders.status == "completed"].merge(custs, on="customer_id").merge(prods, on="product_id")
pt = pd.pivot_table(done, values="order_amount", index="city", columns="category", aggfunc="sum", margins=True, margins_name="Total")
pt_total = float(pt.loc["Total", "Total"])
rep.add("**Pivot table of completed revenue (INR), city x category, with margins**")
rep.add(md_table(pt.round(2), index=True))
rep.add(f"Pivot grand total INR {pt_total:,.2f} vs SQL Q2 total INR {q2.revenue.sum():,.2f} -> "
        f"{'equal' if abs(pt_total - q2.revenue.sum()) < 0.01 else 'DIFFERENT'}.")
assert abs(pt_total - q2.revenue.sum()) < 0.01
city = (done.groupby("city").agg(total=("order_amount", "sum"), mean=("order_amount", "mean"),
                                  count=("order_amount", "count")).reset_index())
rep.add("**Per city (completed orders, named aggregation)**")
rep.add(md_table(city.assign(count=city["count"].astype(int))))

x = done["order_amount"]
q1_, q3_ = x.quantile(.25), x.quantile(.75)
iqr = q3_ - q1_
lo, hi = q1_ - 1.5 * iqr, q3_ + 1.5 * iqr
stats = pd.DataFrame({"statistic": ["mean", "median", "skewness", "Q1", "Q3", "lower fence", "upper fence",
                                      "outliers below lower fence", "outliers above upper fence"],
                      "value": [f"{x.mean():,.2f}", f"{x.median():,.2f}", f"{x.skew():.2f}", f"{q1_:,.2f}", f"{q3_:,.2f}",
                                f"{lo:,.2f}", f"{hi:,.2f}", f"{int((x < lo).sum())}", f"{int((x > hi).sum())}"]})
rep.add("**Completed order_amount statistics**")
rep.add(md_table(stats))
corr = done[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
rep.add("**Correlation matrix (completed orders)**")
rep.add(md_table(corr, index=True, floatfmt="{:.3f}"))
pairs = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
best = pairs.abs().idxmax()
rep.add(f"Largest absolute correlation: {best[0]} vs {best[1]} = {pairs[best]:.3f}.")
rep.add("**Four questions (<=200 words)**\n\n"
        "1. *What is the data about?* Quick-commerce orders in six Indian cities (Apr-Sep 2025): about 11,100 completed orders "
        "across groceries, dairy, snacks, personal care and electronics.\n"
        f"2. *Distributions and statistics?* Order amounts are strongly right-skewed: mean INR {x.mean():,.2f} versus median INR {x.median():,.2f}, "
        f"skewness {x.skew():.2f}, middle half between INR {q1_:,.2f} and INR {q3_:,.2f}.\n"
        f"3. *Outliers or anomalies?* No value lies below the lower fence (INR {lo:,.2f}), but {int((x > hi).sum())} orders exceed the upper fence of "
        f"INR {hi:,.2f}; these are mostly legitimate electronics purchases, not errors. Earlier anomalies (negative delivery times, text amounts, duplicates) were cleaned.\n"
        f"4. *Relationships?* All correlations are weak; the largest in absolute value is {best[0]} with {best[1]} ({pairs[best]:.3f}); "
        f"order_amount with quantity is {corr.loc['order_amount', 'quantity']:+.3f}. Correlation of this size does not establish causation.")
rep.save()
