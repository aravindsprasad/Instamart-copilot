"""Part A validation: asserts the reference checkpoints and prints exactly ALL CHECKPOINTS PASSED."""
import sqlite3
import pandas as pd

con = sqlite3.connect("data/instamart.db")
o = pd.read_sql_query("select * from orders", con)
c = pd.read_sql_query("select * from customers", con)
p = pd.read_sql_query("select * from products", con)
raw = pd.read_csv("data/raw/orders.csv", dtype=str, keep_default_na=False)
comp = o[o.status == "completed"]
canc = o[o.status == "cancelled"]

checks = [
    ("raw order rows", len(raw) == 13084),
    ("raw exact duplicates", int(raw.duplicated().sum()) == 119),
    ("raw pending rows", int((raw.status == "pending").sum()) == 656),
    ("raw distinct order_id", raw.order_id.nunique() == 12309),
    ("tables exist with expected row counts", (len(c), len(p), len(o)) == (1200, 40, 12309)),
    ("clean order rows", len(o) == 12309 and o.order_id.is_unique),
    ("completed / cancelled", (len(comp), len(canc)) == (11110, 1199)),
    ("no pending rows", int((o.status == "pending").sum()) == 0),
    ("no missing order_amount", int(o.order_amount.isna().sum()) == 0),
    ("no missing delivery among completed", int(comp.delivery_minutes.isna().sum()) == 0),
    ("1199 missing delivery among cancelled", int(canc.delivery_minutes.isna().sum()) == 1199),
    ("completed revenue INR 2,998,668.80", abs(comp.order_amount.sum() - 2998668.80) < 0.005),
    ("six cities", sorted(c.city.unique()) == ["Bengaluru", "Chennai", "Delhi", "Hyderabad", "Mumbai", "Pune"]),
    ("city counts", c.city.value_counts().to_dict() == {"Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}),
    ("gender counts", c.gender.value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}),
    ("ages complete and within 18-60", c.age.notna().all() and c.age.between(18, 60).all()),
    ("order_ts stored as YYYY-MM-DD HH:MM:SS text", o.order_ts.str.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}").all()),
]
failed = [n for n, ok in checks if not ok]
for n in failed:
    print("FAILED:", n)
assert not failed, f"{len(failed)} checkpoint(s) failed"
print("ALL CHECKPOINTS PASSED")
