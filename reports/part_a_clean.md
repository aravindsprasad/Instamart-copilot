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
