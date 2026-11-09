### D.5 Tools

| tool | description | parameters |
|---|---|---|
| get_order | Look up an order's status, amount and delivery time in the Instamart database. | order_id |
| search_policy | Search the company policy documents and return the three best matching chunks. | query |
| calculate_refund | Compute the refund in INR from the order amount and the delivery time following the refund policy. | order_amount, delivery_minutes |
| issue_refund | Issue an automatic refund of up to INR 1000 and record it in the ledger. | order_id, refund_inr |
| escalate_to_human | Escalate a refund above INR 1000 to a human support lead and record it in the ledger. | order_id, refund_inr |

`TOOL_MAP` maps each tool name to its Python function; the loop looks up the name the model asks for and calls the function with the arguments dictionary. Refund rules implemented in the tools: only completed orders are refunded (the model path denies other statuses), delivery > 45 min -> 100% of order_amount, otherwise 0, and refunds above INR 1000 go through `escalate_to_human` instead of `issue_refund`. `issue_refund` / `escalate_to_human` append one JSON line (timestamp, tool, run_id, order_id, refund_inr) to `outputs/refund_ledger.jsonl`.

### D.6 ReAct loop - the seven scenarios (also in `outputs/agent_runs.jsonl`)

| scenario id | status | refund INR | expected refund INR | model calls | tokens used | as expected |
|---|---|---|---|---|---|---|
| S1 | REFUND | 128.25 | 128.25 | 5 | 1057 | yes |
| S2 | REFUND | 135.00 | 135.00 | 5 | 1057 | yes |
| S3 | ESCALATE | 1,999.00 | 1,999.00 | 5 | 1044 | yes |
| S4 | DENIED | 0.00 | 0.00 | 4 | 733 | yes |
| S5 | DENIED | 0.00 | 0.00 | 3 | 433 | yes |
| S6 | NOT_FOUND | 0.00 | 0.00 | 2 | 136 | yes |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 0.00 | 6 | 3192 | yes |

S7 (`[LOOP]`) keeps asking for `search_policy`, finishes all six model calls (3,192 estimated tokens, below the default limit of 6000) and ends `FALLBACK_MAX_ITERATIONS` with a human-handoff summary.

### D.7 Stop safely

| run | status | model calls | tokens used | tools executed |
|---|---|---|---|---|
| S7 default (max_iterations=6, token_limit=6000) | FALLBACK_MAX_ITERATIONS | 6 | 3192 | 6 |
| S1 with token_limit=200 | FALLBACK_TOKEN_LIMIT | 3 | 450 | 2 |
| S1 with time_limit_seconds=0 | FALLBACK_TIMEOUT | 0 | 0 | 0 |

- Token-limit demo: the limit was exceeded after model call 3 (450 tokens > 200), the tool that call requested was **not** executed. final_answer: "HANDOFF to human agent: automated run stopped (token limit 200 exceeded). Request: My order ORD-100048 reached me very late. I would like a refund. Steps completed: get_order -> search_policy. Model calls: 3, tokens: 450."
- Timeout demo: `elapsed >= time_limit_seconds` is true at the top of the first iteration (0 >= 0), so the run stops before any model call. final_answer: "HANDOFF to human agent: automated run stopped (time limit of 0 s reached). Request: My order ORD-100048 reached me very late. I would like a refund. Steps completed: no tools. Model calls: 0, tokens: 0."

### D.8 Resumable run (crash after the second tool, then resume)

Simulated crash: *simulated crash after tool #2 (search_policy)*. Checkpoint `state/S1_crash_resume_demo.json` holds run_id, scenario_id, the full message list (6 messages), tokens_used (166), step (2) and tools already executed (get_order, search_policy).

`resume_agent('S1_crash_resume_demo')` loaded the checkpoint and continued: final status **REFUND**, refund INR 128.25, 5 model calls in total.

**Tools executed across the crash and the resume, in order (4 executions, none repeated):** `get_order` -> `search_policy` -> `calculate_refund` -> `issue_refund`.

### D.9 ReAct versus plan-and-execute

Measured with the mock model, my ReAct agent needs 5 model calls for S1 (1,057 estimated tokens), 4 for S4, 3 for S5 (433 tokens) and 2 for S6 (136 tokens). Every call re-sends the whole growing message history, so tokens rise faster than calls. A plan-and-execute design asks the model once for the whole step list (get_order, search_policy, calculate_refund, issue_refund) plus at most one call to write the answer, so it needs about 2 calls and clearly fewer tokens, and the model is not re-invoked after every tool.

Its weakness is the unexpected observation. In S6 (NOT_FOUND) and S5 (a cancelled order) the plan made before seeing the data is wrong: the remaining steps must be skipped or the plan revised, and a blindly executed tail could calculate or even issue a refund for an order that is not refundable. The ReAct agent reacts immediately: it stops after 2 calls for NOT_FOUND and after 3 calls for a cancelled order, and routes a refund above INR 1000 to a human only after seeing the computed amount.

For this refund workflow I would choose ReAct (with the hard limits of D.7): the chains are short (at most 5 calls), money-moving steps depend on earlier observations, and the extra tokens are cheap compared with one wrong refund. Plan-and-execute becomes attractive only for long, predictable workflows.

_(word count: 224)_

Outputs: `outputs/agent_runs.jsonl` (7 lines, one per scenario run), `outputs/agent_demos.jsonl` (the three demonstration runs), `outputs/refund_ledger.jsonl` (4 issued/escalated actions, each tagged with its run_id; the crash-and-resume demo adds one `issue_refund` for S1) and `state/` (9 checkpoint files).
