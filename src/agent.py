"""Part D tasks 5-8: tools, ReAct loop with safe stopping, checkpoints and resume. Offline: uses mock_llm only."""
import json, os, sqlite3, sys, time
from datetime import datetime, timezone
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import mock_llm
import rag

DB, LEDGER, STATE_DIR = "data/instamart.db", "outputs/refund_ledger.jsonl", "state"
SYSTEM_PROMPT = ("You are a refund AGENT for Instamart support. Use the tools to look up the order, check the policy, "
                 "calculate the refund and either issue it or escalate it. Finish with STATUS: short explanation.")
_CURRENT_RUN = {"id": None}


class SimulatedCrash(RuntimeError):
    """Raised on purpose to demonstrate resuming from a checkpoint."""


# ------------------------------------------------------------------ tools
def get_order(order_id):
    with sqlite3.connect(DB) as con:
        row = con.execute("select order_id, status, order_amount, delivery_minutes from orders where order_id = ?", (order_id,)).fetchone()
    if row is None:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1], "order_amount": row[2],
            "delivery_minutes": None if row[3] is None or row[3] != row[3] else row[3]}


def search_policy(query):
    return {"results": rag.retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes):
    """Delivered in more than 45 minutes -> 100% of order_amount, otherwise 0."""
    if delivery_minutes is not None and float(delivery_minutes) > 45:
        return {"refund_inr": round(float(order_amount), 2)}
    return {"refund_inr": 0}


def _ledger(tool, order_id, refund_inr):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    line = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tool": tool,
            "run_id": _CURRENT_RUN["id"], "order_id": order_id, "refund_inr": refund_inr}
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    return {"refund_inr": refund_inr}


def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)


def _spec(name, desc, props, required):
    return {"name": name, "description": desc, "parameters": {"type": "object", "properties": props, "required": required}}


TOOLS_SPEC = [
    _spec("get_order", "Look up an order's status, amount and delivery time in the Instamart database.",
          {"order_id": {"type": "string", "description": "Order id such as ORD-100048"}}, ["order_id"]),
    _spec("search_policy", "Search the company policy documents and return the three best matching chunks.",
          {"query": {"type": "string", "description": "Search text"}}, ["query"]),
    _spec("calculate_refund", "Compute the refund in INR from the order amount and the delivery time following the refund policy.",
          {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}, ["order_amount", "delivery_minutes"]),
    _spec("issue_refund", "Issue an automatic refund of up to INR 1000 and record it in the ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, ["order_id", "refund_inr"]),
    _spec("escalate_to_human", "Escalate a refund above INR 1000 to a human support lead and record it in the ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, ["order_id", "refund_inr"]),
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}


# ------------------------------------------------------------------ checkpoints
def _save(state):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(os.path.join(STATE_DIR, f"{state['run_id']}.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)


def _fallback(state, status, reason):
    done = " -> ".join(s["tool"] for s in state["steps"]) or "no tools"
    answer = (f"HANDOFF to human agent: automated run stopped ({reason}). Request: {state['user_message']} "
              f"Steps completed: {done}. Model calls: {state['llm_calls']}, tokens: {state['tokens_used']}.")
    return {"status": status, "final_answer": answer, "steps": state["steps"], "llm_calls": state["llm_calls"],
            "tokens_used": state["tokens_used"], "run_id": state["run_id"]}


def _loop(state, max_iterations, token_limit, time_limit_seconds, crash_after_tools=None):
    start = time.monotonic()
    _CURRENT_RUN["id"] = state["run_id"]
    while state["llm_calls"] < max_iterations:
        if time.monotonic() - start >= time_limit_seconds:                      # (c) wall-clock limit
            return _fallback(state, "FALLBACK_TIMEOUT", f"time limit of {time_limit_seconds} s reached")
        resp = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += resp.usage.total_tokens
        if state["tokens_used"] > token_limit:                                    # (b) token limit, tool NOT executed
            return _fallback(state, "FALLBACK_TOKEN_LIMIT", f"token limit {token_limit} exceeded")
        msg = resp.choices[0].message
        if not msg.tool_calls:
            answer = msg.content or ""
            return {"status": answer.split(":")[0].strip(), "final_answer": answer, "steps": state["steps"],
                    "llm_calls": state["llm_calls"], "tokens_used": state["tokens_used"], "run_id": state["run_id"]}
        tc = msg.tool_calls[0]
        state["messages"].append({"role": "assistant", "content": None,
                                  "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
        observation = TOOL_MAP[tc.name](**tc.arguments)
        state["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": json.dumps(observation)})
        state["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": observation})
        state["tools_executed"].append(tc.name)
        state["step"] = len(state["steps"])
        _save(state)                                                               # checkpoint after every observation
        if crash_after_tools is not None and len(state["tools_executed"]) == crash_after_tools:
            raise SimulatedCrash(f"simulated crash after tool #{crash_after_tools} ({tc.name})")
    return _fallback(state, "FALLBACK_MAX_ITERATIONS", f"{max_iterations} model calls without a final answer")  # (a)


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tools=None):
    state = {"run_id": run_id or f"{scenario_id}_default", "scenario_id": scenario_id, "user_message": user_message,
             "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}],
             "tokens_used": 0, "step": 0, "tools_executed": [], "steps": [], "llm_calls": 0}
    return _loop(state, max_iterations, token_limit, time_limit_seconds, crash_after_tools)


def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    with open(os.path.join(STATE_DIR, f"{run_id}.json"), encoding="utf-8") as f:
        state = json.load(f)
    return _loop(state, max_iterations, token_limit, time_limit_seconds)


def refund_amount(result):
    for s in reversed(result["steps"]):
        if s["tool"] in ("issue_refund", "escalate_to_human"):
            return float(s["observation"]["refund_inr"])
    return 0.0
