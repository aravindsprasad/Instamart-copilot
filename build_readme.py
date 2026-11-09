"""Assemble README.md from the hand-written sections below and the report fragments written by the Part A-D scripts (reports/*.md)."""
def frag(name):
    return open(f"reports/{name}.md", encoding="utf-8").read().strip()

OVERVIEW = """# Instamart Customer Intelligence & Support Copilot

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
"""

REPRODUCE = """## Reproduce

Run from the repository root (Python 3.11 or newer; SQLite 3.25 or newer). The commands are listed in order.

1. `python -m venv .venv`
2. `source .venv/bin/activate` (on Windows: `.venv\\Scripts\\activate`)
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
"""

PART_B_NOTES = """### B.9 Leakage statement

- **No encoder, scaler or resampler saw test rows.** The `StandardScaler` and `OneHotEncoder(drop="first")` live inside a scikit-learn `Pipeline` and are fitted on training rows only
  (inside `GridSearchCV` only on each fold's training part); the oversampler duplicated lapsed rows of the training set only, and the test set stayed unchanged.
- **No data from 2025-09-01 onwards entered the features:** every feature is computed from orders dated before the cutoff (asserted in code) plus the clean customer table.
- **The segment labels from B.3 were not used as features.**
- The Premium flag (B.3) is a post-hoc whole-window label, so the retention list is illustrative and not a leakage-free evaluation (see Limitations and Next Steps).
"""

LIMITS = """## Limitations and Next Steps

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
"""

parts = [OVERVIEW, REPRODUCE,
         "## Part A\n\n" + frag("part_a_clean") + "\n\n" + frag("part_a_sql"),
         "## Part B\n\n" + frag("part_b_segments") + "\n\n" + frag("part_b_lapse") + "\n\n" + PART_B_NOTES,
         "## Part C\n\n" + frag("part_c_triage"),
         "## Part D\n\n" + frag("part_d_rag") + "\n\n" + frag("part_d_agent"),
         LIMITS]
with open("README.md", "w", encoding="utf-8") as f:
    f.write("\n\n".join(p.strip() for p in parts) + "\n")
print("README.md written")
