### C.1 Input guardrails

Empty or whitespace-only tickets -> `skipped_empty`: 4. Tickets with est_tokens > 400 -> `flagged_long`: 4 (each estimates to [734] tokens). Neither kind is sent to the model.

### C.2 Redaction

| type | substitutions | reference |
|---|---|---|
| email | 55 | 55 |
| phone | 91 | 91 |
| order id | 132 | 132 |
| total | 278 | 278 |

Re-applying the three patterns to the redacted texts finds **0** matches (reference 0). Patterns are applied in the order email, phone, order id; the user message sent to the model is exactly the redacted ticket text.

### C.3 Prompts

Zero-shot system prompt:

```
You CLASSIFY customer support tickets. Allowed categories: billing, delivery, technical, fraud. Reply with JSON only, exactly of the form {"category": "...", "reason": "..."}.
```

Few-shot system prompt:

```
You CLASSIFY customer support tickets. Allowed categories: billing, delivery, technical, fraud. Reply with JSON only, exactly of the form {"category": "...", "reason": "..."}.
EXAMPLES:
Ticket: I was charged twice for [ORDER_ID]. Please refund the extra payment. -> {"category": "billing", "reason": "double charge"}
Ticket: Someone hacked my account and used my wallet balance. -> {"category": "fraud", "reason": "unauthorised account use"}
Ticket: The app freezes when I apply a coupon. -> {"category": "technical", "reason": "app fault"}
Ticket: My order has not arrived and the rider is not picking up. -> {"category": "delivery", "reason": "missing delivery"}
```

Prompt length: zero-shot 30 estimated tokens, few-shot 121 estimated tokens.

### C.4 Resilient calls

Each full run (after `mock_llm.reset()`) raised exactly 19 TransientLLMErrors (zero-shot) and 19 (few-shot) over the 156 eligible tickets; all were recovered by a retry (max 3 retries, wait = 0.05 x 2^n x uniform(0.9, 1.15) s) and no ticket ended with a permanent error. Every retry was retry number 0 and its wait lay in [0.0451, 0.0572] s (allowed 0.045 - 0.0575). One log line per retry is saved in `outputs/retry_log.txt`.

### C.5 Output validation (one full run over the 156 eligible tickets)

| status | zero-shot | few-shot | reference |
|---|---|---|---|
| classified_json | 135 | 135 | 135 |
| classified_keyword | 12 | 12 | 12 |
| flagged_invalid_output | 9 | 9 | 9 |

A reply that is valid JSON with a category outside the list (the mock's `shipping`) or a chatty reply with zero or several category words is `flagged_invalid_output` and goes to human review; the pipeline never guesses. Keyword extraction uses `re.findall(r"[a-z]+", text.lower())`, so `billing-related` yields the words `billing` and `related`.

### C.6 Zero-shot versus few-shot

| run | tickets classified | correct | accuracy % | total tokens | tokens per classified ticket |
|---|---|---|---|---|---|
| zero-shot | 147 | 126 | 85.7 | 9,088 | 61.8 |
| few-shot | 147 | 147 | 100.0 | 23,284 | 158.4 |

**Confusion table, zero-shot** (rows = true category, columns = predicted)

| true \ predicted | billing | delivery | technical | fraud |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| technical | 10 | 0 | 29 | 0 |
| fraud | 11 | 0 | 0 | 23 |

**When are few-shot tokens worth it?** The few-shot prompt costs 14,196 more tokens over the run (96.6 more per classified ticket, 156% more) but lifts accuracy from 85.7% to 100.0%: the zero-shot run misclassified 21 tickets (all fraud or technical tickets predicted as billing), which are exactly the cases where a mistake is expensive (a fraud report not reaching the security team). For a high-stakes category like fraud the extra ~97 tokens per ticket are clearly worth paying; for a low-stakes triage where billing/technical confusion is cheap to fix, or at very high volume with a tight budget, the cheaper zero-shot prompt may be enough. With the mock model the gain comes from the order in which keyword groups are checked.

### C.7 Token budget and routing

Zero-shot run with TOKEN_BUDGET = 6000: the running total reached the budget after 103 model calls (6,018 tokens) and **53 tickets ended `budget_exhausted`** (no call made, routed to human review). Saved as `outputs/triage_budget_demo.csv`.

`outputs/triage_results.csv` comes from a few-shot run with TOKEN_BUDGET = 100000 (no ticket exhausted: 0). Routing rules: skipped_empty / flagged_long / flagged_invalid_output / budget_exhausted -> `human_review` (NORMAL); else fraud -> `security_team` (HIGH); else Premium customer -> `premium_desk` (HIGH); else `support_queue` (NORMAL). A customer with no row in the segment file counts as not premium.

| route | priority | tickets |
|---|---|---|
| human_review | NORMAL | 17 |
| premium_desk | HIGH | 11 |
| security_team | HIGH | 34 |
| support_queue | NORMAL | 102 |

| status | tickets |
|---|---|
| classified_json | 135 |
| classified_keyword | 12 |
| flagged_invalid_output | 9 |
| skipped_empty | 4 |
| flagged_long | 4 |

**Rule-validated vs human:** 147 tickets were classified and passed output validation (routed automatically); 17 went to human review (4 empty, 4 too long, 9 invalid model output).

Privacy assertion: none of the 571 user messages sent to the model across all four runs contains an email, a phone number or an order id (asserted in code); no empty or long ticket triggered a model call (model calls per run = 156 eligible tickets).
