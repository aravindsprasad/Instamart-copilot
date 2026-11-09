"""Part C: support-ticket triage with guardrails (input checks, redaction, retries, output validation, budget, routing)."""
import json, os, random, re, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pandas as pd
import mock_llm
from mock_llm import TransientLLMError, est_tokens
from util import Report, md_table

rep = Report("part_c_triage")
MAX_INPUT_TOKENS = 400
MAX_RETRIES, BASE_DELAY = 3, 0.05
ALLOWED = ["billing", "delivery", "technical", "fraud"]
RETRY_LOG = []                                   # one line per retry, written to outputs/retry_log.txt
random.seed(42)                                  # reproducible jitter

tickets = pd.read_csv("data/raw/tickets.csv", dtype=str, keep_default_na=False)
assert len(tickets) == 164

# ------------------------------------------------------------------ redaction
PATTERNS = [("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
            ("PHONE", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
            ("ORDER_ID", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]

def redact(text, counts=None):
    for name, pat, ph in PATTERNS:               # applied in this exact order
        text, n = pat.subn(ph, text)
        if counts is not None:
            counts[name] += n
    return text

# ------------------------------------------------------------------ prompts
ZERO_SHOT = ("You CLASSIFY customer support tickets. Allowed categories: billing, delivery, technical, fraud. "
             'Reply with JSON only, exactly of the form {"category": "...", "reason": "..."}.')
FEW_SHOT = ("You CLASSIFY customer support tickets. Allowed categories: billing, delivery, technical, fraud. "
            'Reply with JSON only, exactly of the form {"category": "...", "reason": "..."}.\n'
            "EXAMPLES:\n"
            'Ticket: I was charged twice for [ORDER_ID]. Please refund the extra payment. -> {"category": "billing", "reason": "double charge"}\n'
            'Ticket: Someone hacked my account and used my wallet balance. -> {"category": "fraud", "reason": "unauthorised account use"}\n'
            'Ticket: The app freezes when I apply a coupon. -> {"category": "technical", "reason": "app fault"}\n'
            'Ticket: My order has not arrived and the rider is not picking up. -> {"category": "delivery", "reason": "missing delivery"}')
assert "CLASSIFY" in ZERO_SHOT and "EXAMPLES:" not in ZERO_SHOT and "EXAMPLES:" in FEW_SHOT and "CLASSIFY" in FEW_SHOT

# ------------------------------------------------------------------ resilient call
def call_with_retry(messages, ticket_id):
    for n in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except TransientLLMError as e:
            if n == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** n * random.uniform(0.9, 1.15)
            RETRY_LOG.append((ticket_id, n, wait))
            RETRY_LOG_LINES.append(f"[{RUN}] retry ticket={ticket_id} retry_number={n} wait={wait:.4f}s error={e}")
            time.sleep(wait)

RETRY_LOG_LINES, RUN = [], ""
SENT_PROMPTS = []                                # every user message that goes to the model (for the privacy assertion)

def validate(text):
    """Return (status, category). Never guesses."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        words = set(re.findall(r"[a-z]+", (text or "").lower()))
        hits = [c for c in ALLOWED if c in words]
        return ("classified_keyword", hits[0]) if len(hits) == 1 else ("flagged_invalid_output", "")
    if isinstance(obj, dict) and obj.get("category") in ALLOWED:
        return "classified_json", obj["category"]
    return "flagged_invalid_output", ""

def run_pipeline(system_prompt, label, token_budget=None):
    global RUN
    RUN = label
    mock_llm.reset()
    RETRY_LOG.clear()
    counts_before = len(RETRY_LOG_LINES)
    errors_seen = 0
    total_tokens, rows = 0, []
    for t in tickets.itertuples(index=False):
        text = t.text
        rec = dict(ticket_id=t.ticket_id, customer_id=t.customer_id, true_category=t.true_category, status="", category="", tokens_used=0)
        if not text.strip():
            rec["status"] = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:
            rec["status"] = "flagged_long"
        elif token_budget is not None and total_tokens >= token_budget:
            rec["status"] = "budget_exhausted"
        else:
            user = redact(text)
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]
            SENT_PROMPTS.append(user)
            before = len(RETRY_LOG)
            resp = call_with_retry(messages, t.ticket_id)
            rec["tokens_used"] = resp.usage.total_tokens
            total_tokens += resp.usage.total_tokens
            rec["status"], rec["category"] = validate(resp.choices[0].message.content)
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.attrs["total_tokens"] = total_tokens
    df.attrs["retries"] = len(RETRY_LOG)
    df.attrs["retry_waits"] = [w for _, n, w in RETRY_LOG]
    return df

# ------------------------------------------------------------------ guardrail + redaction report
rep.add("### C.1 Input guardrails")
n_empty = int((~tickets.text.str.strip().astype(bool)).sum())
n_long = int(tickets.text.map(lambda x: est_tokens(x) > MAX_INPUT_TOKENS).sum())
long_tok = sorted({est_tokens(x) for x in tickets.text if est_tokens(x) > MAX_INPUT_TOKENS})
rep.add(f"Empty or whitespace-only tickets -> `skipped_empty`: {n_empty}. Tickets with est_tokens > {MAX_INPUT_TOKENS} -> `flagged_long`: {n_long} "
        f"(each estimates to {long_tok} tokens). Neither kind is sent to the model.")
rep.add("### C.2 Redaction")
counts = {"EMAIL": 0, "PHONE": 0, "ORDER_ID": 0}
red = [redact(x, counts) for x in tickets.text]
left = sum(len(p.findall(x)) for x in red for _, p, _ in PATTERNS)
rep.add(md_table(pd.DataFrame({"type": ["email", "phone", "order id", "total"], "substitutions": [counts["EMAIL"], counts["PHONE"], counts["ORDER_ID"], sum(counts.values())],
                               "reference": [55, 91, 132, 278]})))
rep.add(f"Re-applying the three patterns to the redacted texts finds **{left}** matches (reference 0). Patterns are applied in the order email, phone, order id; "
        "the user message sent to the model is exactly the redacted ticket text.")
assert counts == {"EMAIL": 55, "PHONE": 91, "ORDER_ID": 132} and left == 0

# ------------------------------------------------------------------ runs
rep.add("### C.3 Prompts")
rep.add("Zero-shot system prompt:\n\n```\n" + ZERO_SHOT + "\n```\n\nFew-shot system prompt:\n\n```\n" + FEW_SHOT + "\n```")
rep.add(f"Prompt length: zero-shot {est_tokens(ZERO_SHOT)} estimated tokens, few-shot {est_tokens(FEW_SHOT)} estimated tokens.")

zs = run_pipeline(ZERO_SHOT, "zero-shot")
fs = run_pipeline(FEW_SHOT, "few-shot")

def summarize(df):
    cl = df[df.status.isin(["classified_json", "classified_keyword"])]
    return dict(classified=len(cl), correct=int((cl.category == cl.true_category).sum()),
                acc=(cl.category == cl.true_category).mean() * 100, tokens=df.attrs["total_tokens"],
                per=df.attrs["total_tokens"] / len(cl), counts=df.status.value_counts().to_dict(), retries=df.attrs["retries"])
Sz, Sf = summarize(zs), summarize(fs)

rep.add("### C.4 Resilient calls")
all_waits = zs.attrs["retry_waits"] + fs.attrs["retry_waits"]
rep.add(f"Each full run (after `mock_llm.reset()`) raised exactly {Sz['retries']} TransientLLMErrors (zero-shot) and {Sf['retries']} (few-shot) over the 156 eligible tickets; "
        f"all were recovered by a retry (max {MAX_RETRIES} retries, wait = 0.05 x 2^n x uniform(0.9, 1.15) s) and no ticket ended with a permanent error. "
        f"Every retry was retry number 0 and its wait lay in [{min(all_waits):.4f}, {max(all_waits):.4f}] s (allowed 0.045 - 0.0575). "
        "One log line per retry is saved in `outputs/retry_log.txt`.")
assert Sz["retries"] == 19 and Sf["retries"] == 19 and all(0.045 <= w <= 0.0575 for w in all_waits)

rep.add("### C.5 Output validation (one full run over the 156 eligible tickets)")
rep.add(md_table(pd.DataFrame({"status": ["classified_json", "classified_keyword", "flagged_invalid_output"],
                               "zero-shot": [Sz["counts"].get(s, 0) for s in ["classified_json", "classified_keyword", "flagged_invalid_output"]],
                               "few-shot": [Sf["counts"].get(s, 0) for s in ["classified_json", "classified_keyword", "flagged_invalid_output"]],
                               "reference": [135, 12, 9]})))
assert [Sz["counts"].get(s, 0) for s in ["classified_json", "classified_keyword", "flagged_invalid_output"]] == [135, 12, 9]
assert [Sf["counts"].get(s, 0) for s in ["classified_json", "classified_keyword", "flagged_invalid_output"]] == [135, 12, 9]
rep.add("A reply that is valid JSON with a category outside the list (the mock's `shipping`) or a chatty reply with zero or several category words is "
        "`flagged_invalid_output` and goes to human review; the pipeline never guesses. Keyword extraction uses `re.findall(r\"[a-z]+\", text.lower())`, so "
        "`billing-related` yields the words `billing` and `related`.")

rep.add("### C.6 Zero-shot versus few-shot")
rep.add(md_table(pd.DataFrame({"run": ["zero-shot", "few-shot"], "tickets classified": [Sz["classified"], Sf["classified"]],
                               "correct": [Sz["correct"], Sf["correct"]], "accuracy %": [f"{Sz['acc']:.1f}", f"{Sf['acc']:.1f}"],
                               "total tokens": [f"{Sz['tokens']:,}", f"{Sf['tokens']:,}"], "tokens per classified ticket": [f"{Sz['per']:.1f}", f"{Sf['per']:.1f}"]})))
clz = zs[zs.status.isin(["classified_json", "classified_keyword"])]
conf = pd.crosstab(clz.true_category, clz.category).reindex(index=ALLOWED, columns=ALLOWED, fill_value=0)
conf.index.name = "true \\ predicted"
rep.add("**Confusion table, zero-shot** (rows = true category, columns = predicted)")
rep.add(md_table(conf, index=True, floatfmt="{:.0f}"))
extra = Sf["tokens"] - Sz["tokens"]
errs = clz[clz.category != clz.true_category]
rep.add(f"**When are few-shot tokens worth it?** The few-shot prompt costs {extra:,} more tokens over the run ({Sf['per'] - Sz['per']:.1f} more per classified ticket, "
        f"{(Sf['tokens'] / Sz['tokens'] - 1) * 100:.0f}% more) but lifts accuracy from {Sz['acc']:.1f}% to {Sf['acc']:.1f}%: the zero-shot run misclassified {len(errs)} tickets "
        f"(all fraud or technical tickets predicted as billing), which are exactly the cases where a mistake is expensive (a fraud report not reaching the security team). "
        f"For a high-stakes category like fraud the extra ~{Sf['per'] - Sz['per']:.0f} tokens per ticket are clearly worth paying; for a low-stakes triage where billing/technical confusion "
        "is cheap to fix, or at very high volume with a tight budget, the cheaper zero-shot prompt may be enough. With the mock model the gain comes from the order in which keyword groups are checked.")
assert abs(Sz["acc"] - 85.7) <= .5 and abs(Sf["acc"] - 100) <= .5 and Sz["classified"] == 147

# ------------------------------------------------------------------ budget demo
rep.add("### C.7 Token budget and routing")
BUDGET_DEMO = 6000
bd = run_pipeline(ZERO_SHOT, "budget-demo", token_budget=BUDGET_DEMO)
n_exh = int((bd.status == "budget_exhausted").sum())
bd_out = bd.copy()
bd_out["route"] = "human_review"
bd[["ticket_id", "customer_id", "status", "category", "tokens_used"]].to_csv("outputs/triage_budget_demo.csv", index=False)
rep.add(f"Zero-shot run with TOKEN_BUDGET = {BUDGET_DEMO}: the running total reached the budget after {int((bd.tokens_used > 0).sum())} model calls "
        f"({bd.attrs['total_tokens']:,} tokens) and **{n_exh} tickets ended `budget_exhausted`** (no call made, routed to human review). "
        "Saved as `outputs/triage_budget_demo.csv`.")
assert n_exh >= 1

# ------------------------------------------------------------------ final routing
final = run_pipeline(FEW_SHOT, "final-few-shot", token_budget=100000)
seg = pd.read_csv("outputs/customer_segments.csv").set_index("customer_id")
HUMAN = {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"}
def route(r):
    if r.status in HUMAN:
        return "human_review", "NORMAL"
    if r.category == "fraud":
        return "security_team", "HIGH"
    if r.customer_id in seg.index and bool(seg.loc[r.customer_id, "is_premium"]):
        return "premium_desk", "HIGH"
    return "support_queue", "NORMAL"
rr = final.apply(route, axis=1, result_type="expand")
final["route"], final["priority"] = rr[0], rr[1]
final["segment_name"] = final.customer_id.map(lambda c: seg.loc[c, "segment_name"] if c in seg.index else "")
final[["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]].to_csv("outputs/triage_results.csv", index=False)
assert len(final) == 164
rep.add("`outputs/triage_results.csv` comes from a few-shot run with TOKEN_BUDGET = 100000 (no ticket exhausted: "
        f"{int((final.status == 'budget_exhausted').sum())}). Routing rules: skipped_empty / flagged_long / flagged_invalid_output / budget_exhausted -> `human_review` (NORMAL); "
        "else fraud -> `security_team` (HIGH); else Premium customer -> `premium_desk` (HIGH); else `support_queue` (NORMAL). "
        "A customer with no row in the segment file counts as not premium.")
rep.add(md_table(final.groupby(["route", "priority"]).size().reset_index(name="tickets")))
rep.add(md_table(final.status.value_counts().rename_axis("status").reset_index(name="tickets")))
rule_validated = int(final.status.isin(["classified_json", "classified_keyword"]).sum())
rep.add(f"**Rule-validated vs human:** {rule_validated} tickets were classified and passed output validation (routed automatically); "
        f"{int(final.route.eq('human_review').sum())} went to human review ({int((final.status == 'skipped_empty').sum())} empty, {int((final.status == 'flagged_long').sum())} too long, "
        f"{int((final.status == 'flagged_invalid_output').sum())} invalid model output).")

# privacy assertion over every prompt ever sent
pii = [p for _, p, _ in PATTERNS]
assert all(not p.search(u) for u in SENT_PROMPTS for p in pii), "PII reached the model!"
rep.add(f"Privacy assertion: none of the {len(SENT_PROMPTS)} user messages sent to the model across all four runs contains an email, a phone number or an order id (asserted in code); "
        "no empty or long ticket triggered a model call (model calls per run = 156 eligible tickets).")
with open("outputs/retry_log.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(RETRY_LOG_LINES) + "\n")
rep.save()
