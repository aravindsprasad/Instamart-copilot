"""Part A tasks 2-5: audit the raw tables, clean customers and orders, store CSV + SQLite."""
import os, sqlite3
import numpy as np
import pandas as pd
from util import Report, md_table

RAW, CLEAN, DB = "data/raw", "data/clean", "data/instamart.db"
os.makedirs(CLEAN, exist_ok=True)
rep = Report("part_a_clean")

# ------------------------------------------------------------------ audit
rep.add("### A.1 Audit of the raw data")
orders_raw = pd.read_csv(f"{RAW}/orders.csv")
customers_raw = pd.read_csv(f"{RAW}/customers.csv")
products_raw = pd.read_csv(f"{RAW}/products.csv")
# string versions (nothing converted to NaN) for exact counting of raw spellings / unparseable entries
o_str = pd.read_csv(f"{RAW}/orders.csv", dtype=str, keep_default_na=False)
c_str = pd.read_csv(f"{RAW}/customers.csv", dtype=str, keep_default_na=False)

for name, df in [("orders.csv", orders_raw), ("customers.csv", customers_raw), ("products.csv", products_raw)]:
    rep.add(f"**{name}** - shape {df.shape[0]:,} rows x {df.shape[1]} columns")
    summary = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum(),
                            "missing_%": (df.isna().mean() * 100).round(2)}).reset_index(names="column")
    rep.add(md_table(summary))
rep.add("_Note: `pd.read_csv` treats the literal text `N/A` as missing, so the `order_amount` missing count above "
        "includes the `N/A` entries; the exact non-numeric count below is taken from an all-string read._")

amt_num = pd.to_numeric(o_str["order_amount"], errors="coerce")
dl = pd.to_numeric(o_str["delivery_minutes"], errors="coerce")
age = pd.to_numeric(c_str["age"], errors="coerce")
audit = {
    "orders.csv rows": len(o_str),
    "exact duplicate rows": int(o_str.duplicated().sum()),
    "rows with status pending": int((o_str["status"] == "pending").sum()),
    "distinct order_id": int(o_str["order_id"].nunique()),
    "order_amount not convertible to a number": int(amt_num.isna().sum()),
    "negative delivery_minutes": int((dl < 0).sum()),
    "blank delivery_minutes": int((o_str["delivery_minutes"] == "").sum()),
    "customers: blank age": int(age.isna().sum()),
    "customers: age outside 10-100": int(((age < 10) | (age > 100)).sum()),
    "customers: distinct raw city spellings": int(c_str["city"].nunique()),
    "customers: distinct non-blank raw gender spellings": int(c_str.loc[c_str["gender"] != "", "gender"].nunique()),
    "customers: blank gender": int((c_str["gender"] == "").sum()),
}
reference = [13084, 119, 656, 12309, 120, 106, 2541, 61, 6, 24, 10, 30]
tbl = pd.DataFrame({"audit figure": list(audit), "value": [f"{v:,}" for v in audit.values()],
                    "reference": [f"{r:,}" for r in reference],
                    "match": ["yes" if v == r else "NO" for v, r in zip(audit.values(), reference)]})
rep.add("**Audit figures**")
rep.add(md_table(tbl))
assert list(audit.values()) == reference, "audit figures differ from reference"

# ------------------------------------------------------------------ clean customers
rep.add("### A.2 Cleaning rules - customers")
cust = customers_raw.copy()
cust["city"] = cust["city"].astype(str).str.strip().str.title()
changed_city = int((cust["city"] != customers_raw["city"]).sum())
gmap = {"male": "male", "m": "male", "female": "female", "f": "female"}
cust["gender"] = cust["gender"].fillna("").astype(str).str.strip().str.lower().map(gmap).fillna("unknown")
n_gender_changed = int((c_str["gender"] != cust["gender"]).sum())
bad_age = ((cust["age"] < 10) | (cust["age"] > 100))
n_bad_age, n_blank_age = int(bad_age.sum()), int(cust["age"].isna().sum())
cust.loc[bad_age, "age"] = np.nan
cust["age"] = cust["age"].fillna(cust.groupby("city")["age"].transform("median"))
cust["signup_date"] = pd.to_datetime(cust["signup_date"])
assert cust["city"].nunique() == 6 and cust["age"].isna().sum() == 0 and cust["age"].between(18, 60).all()
rep.add(md_table(pd.DataFrame({
    "rule": ["(a) city: strip + title case -> 6 cities", "(b) gender: map to male / female / unknown (blank -> unknown)",
             "(c) ages outside 10-100 set to missing", "(c) missing ages (blank + invalid) filled with city median (groupby + transform)",
             "(d) signup_date -> datetime"],
    "values changed": [f"{changed_city} city strings", f"{n_gender_changed} gender strings rewritten "
                       f"({int((cust['gender'] == 'unknown').sum())} blank -> unknown)",
                       f"{n_bad_age} ages", f"{n_bad_age + n_blank_age} ages ({n_blank_age} blank + {n_bad_age} invalid)",
                       f"{len(cust)} dates"]})))
rep.add("City counts after cleaning: " + ", ".join(f"{k} {v}" for k, v in cust["city"].value_counts().items()))
rep.add("Gender counts after cleaning: " + ", ".join(f"{k} {v}" for k, v in cust["gender"].value_counts().items()))
rep.add(f"Missing ages: {int(cust['age'].isna().sum())}; age range {cust['age'].min():g}-{cust['age'].max():g}.")

# ------------------------------------------------------------------ clean orders
rep.add("### A.3 Cleaning rules - orders (applied in this order)")
o = orders_raw.copy()
o["order_amount"] = o_str["order_amount"]      # all-string amounts so 'N/A' and 'error' are both visible
steps = []

o["order_ts"] = pd.to_datetime(o["order_ts"])
steps.append(("1. order_ts -> datetime", f"{len(o):,} values converted", f"{len(o):,}"))

before = len(o)
o = o.drop_duplicates()
steps.append(("2. drop exact duplicate rows", f"{before - len(o)} rows removed", f"{len(o):,}"))

before = len(o)
o = o.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last").reset_index(drop=True)
steps.append(("3. keep latest order_ts per order_id", f"{before - len(o)} rows removed (earlier 'pending' records)", f"{len(o):,}"))

o["order_amount"] = pd.to_numeric(o["order_amount"], errors="coerce")
n_missing_amt = int(o["order_amount"].isna().sum())
prod = products_raw.set_index("product_id")["price_inr"]
fill = (o["product_id"].map(prod) * o["quantity"] * (1 - o["discount_pct"] / 100)).round(2)
o["order_amount"] = o["order_amount"].fillna(fill)
steps.append(("4. order_amount -> numeric, fill from price x qty x (1 - disc/100)", f"{n_missing_amt} values filled", f"{len(o):,}"))

o["delivery_minutes"] = pd.to_numeric(o["delivery_minutes"], errors="coerce")
neg = o["delivery_minutes"] < 0
n_neg = int(neg.sum())
o.loc[neg, "delivery_minutes"] = np.nan
o["category"] = o["product_id"].map(products_raw.set_index("product_id")["category"])
comp = o["status"] == "completed"
n_fill_dl = int((comp & o["delivery_minutes"].isna()).sum())
med = o["delivery_minutes"].where(comp).groupby(o["category"]).transform("median")
o.loc[comp, "delivery_minutes"] = o.loc[comp, "delivery_minutes"].fillna(med[comp])
steps.append(("5. negative delivery_minutes -> missing; completed filled with category median; cancelled left missing",
              f"{n_neg} negatives set missing; {n_fill_dl} completed values filled; "
              f"{int((~comp & o['delivery_minutes'].isna()).sum())} cancelled left missing", f"{len(o):,}"))
o = o.drop(columns="category")
rep.add(md_table(pd.DataFrame(steps, columns=["rule", "values/rows changed", "rows after"])))

res = {
    "rows": len(o), "completed": int(comp.sum()), "cancelled": int((o.status == "cancelled").sum()),
    "pending": int((o.status == "pending").sum()), "missing amount": int(o.order_amount.isna().sum()),
    "missing delivery (completed)": int(o.loc[comp, "delivery_minutes"].isna().sum()),
    "missing delivery (cancelled)": int(o.loc[o.status == "cancelled", "delivery_minutes"].isna().sum()),
    "completed revenue INR": round(float(o.loc[comp, "order_amount"].sum()), 2)}
ref = [12309, 11110, 1199, 0, 0, 0, 1199, 2998668.80]
fmt = lambda v: f"{v:,.2f}" if isinstance(v, float) else f"{v:,}"
rep.add(md_table(pd.DataFrame({"check": list(res), "value": [fmt(v) for v in res.values()],
                               "reference": [fmt(r) for r in ref],
                               "match": ["yes" if abs(v - r) < 0.005 else "NO" for v, r in zip(res.values(), ref)]})))
assert all(abs(v - r) < 0.005 for v, r in zip(res.values(), ref)), "order checkpoints differ"

# ------------------------------------------------------------------ store
orders_out = o.copy()
orders_out["order_ts"] = orders_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
cust_out = cust.copy()
cust_out["signup_date"] = cust_out["signup_date"].dt.strftime("%Y-%m-%d")
orders_out.to_csv(f"{CLEAN}/orders.csv", index=False)
cust_out.to_csv(f"{CLEAN}/customers.csv", index=False)
products_raw.to_csv(f"{CLEAN}/products.csv", index=False)
if os.path.exists(DB):
    os.remove(DB)
with sqlite3.connect(DB) as con:
    cust_out.to_sql("customers", con, index=False)
    products_raw.to_sql("products", con, index=False)
    orders_out.to_sql("orders", con, index=False)
rep.add("### A.4 Storage")
rep.add(f"Wrote `data/clean/customers.csv` ({len(cust_out):,} rows), `products.csv` ({len(products_raw)} rows), "
        f"`orders.csv` ({len(orders_out):,} rows) and the same three tables to `data/instamart.db` "
        "(`order_ts` stored as text `YYYY-MM-DD HH:MM:SS`).")
rep.save()
