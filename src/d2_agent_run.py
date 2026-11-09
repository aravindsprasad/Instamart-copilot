"""Part D tasks 5-8: run the 7 scenarios, stop-safely demos, crash-and-resume demo, ReAct vs plan-and-execute."""
import glob, json, os, shutil
import pandas as pd
import agent
from agent import run_agent, resume_agent, refund_amount, SimulatedCrash
from util import Report, md_table

rep = Report("part_d_agent")
# clean outputs so the run is reproducible
for p in ["outputs/agent_runs.jsonl", "outputs/agent_demos.jsonl", "outputs/refund_ledger.jsonl"]:
    if os.path.exists(p):
        os.remove(p)
shutil.rmtree("state", ignore_errors=True)
os.makedirs("state", exist_ok=True)

rep.add("### D.5 Tools")
rep.add(md_table(pd.DataFrame([(t["name"], t["description"], ", ".join(t["parameters"]["properties"])) for t in agent.TOOLS_SPEC],
                              columns=["tool", "description", "parameters"])))
rep.add("`TOOL_MAP` maps each tool name to its Python function; the loop looks up the name the model asks for and calls the function with the arguments dictionary. "
        "Refund rules implemented in the tools: only completed orders are refunded (the model path denies other statuses), delivery > 45 min -> 100% of order_amount, "
        "otherwise 0, and refunds above INR 1000 go through `escalate_to_human` instead of `issue_refund`. `issue_refund` / `escalate_to_human` append one JSON line "
        "(timestamp, tool, run_id, order_id, refund_inr) to `outputs/refund_ledger.jsonl`.")

# ------------------------------------------------------------------ 7 scenarios
sc = pd.read_csv("data/raw/agent_scenarios.csv", keep_default_na=False)
rows, results = [], {}
with open("outputs/agent_runs.jsonl", "a", encoding="utf-8") as f:
    for r in sc.itertuples():
        res = run_agent(r.message, r.scenario_id)
        results[r.scenario_id] = res
        amt = refund_amount(res)
        f.write(json.dumps({"scenario_id": r.scenario_id, "status": res["status"], "final_answer": res["final_answer"],
                            "llm_calls": res["llm_calls"], "tokens_used": res["tokens_used"], "steps": res["steps"]}) + "\n")
        exp = float(r.expected_refund_inr)
        ok = (res["status"] == r.expected_outcome or (r.expected_outcome == "FALLBACK" and res["status"] == "FALLBACK_MAX_ITERATIONS")) \
            and abs(amt - exp) <= 0.01
        rows.append((r.scenario_id, res["status"], f"{amt:,.2f}", f"{exp:,.2f}", res["llm_calls"], res["tokens_used"], "yes" if ok else "NO"))
        assert ok, (r.scenario_id, res["status"], amt, exp)
rep.add("### D.6 ReAct loop - the seven scenarios (also in `outputs/agent_runs.jsonl`)")
rep.add(md_table(pd.DataFrame(rows, columns=["scenario id", "status", "refund INR", "expected refund INR", "model calls", "tokens used", "as expected"])))
assert results["S1"]["llm_calls"] == 5 and results["S5"]["llm_calls"] == 3 and results["S6"]["llm_calls"] == 2
assert results["S7"]["status"] == "FALLBACK_MAX_ITERATIONS" and results["S7"]["llm_calls"] == 6
rep.add(f"S7 (`[LOOP]`) keeps asking for `search_policy`, finishes all six model calls ({results['S7']['tokens_used']:,} estimated tokens, below the default limit of 6000) and ends "
        "`FALLBACK_MAX_ITERATIONS` with a human-handoff summary.")

# ------------------------------------------------------------------ stop safely demos
rep.add("### D.7 Stop safely")
s1_msg = sc.loc[sc.scenario_id == "S1", "message"].iloc[0]
demo_a = run_agent(s1_msg, "S1", token_limit=200, run_id="S1_token_limit_demo")
demo_b = run_agent(s1_msg, "S1", time_limit_seconds=0, run_id="S1_timeout_demo")
assert demo_a["status"] == "FALLBACK_TOKEN_LIMIT" and demo_b["status"] == "FALLBACK_TIMEOUT" and demo_b["llm_calls"] == 0
rep.add(md_table(pd.DataFrame([
    ("S7 default (max_iterations=6, token_limit=6000)", results["S7"]["status"], results["S7"]["llm_calls"], results["S7"]["tokens_used"], len(results["S7"]["steps"])),
    ("S1 with token_limit=200", demo_a["status"], demo_a["llm_calls"], demo_a["tokens_used"], len(demo_a["steps"])),
    ("S1 with time_limit_seconds=0", demo_b["status"], demo_b["llm_calls"], demo_b["tokens_used"], len(demo_b["steps"]))],
    columns=["run", "status", "model calls", "tokens used", "tools executed"])))
rep.add(f"- Token-limit demo: the limit was exceeded after model call {demo_a['llm_calls']} ({demo_a['tokens_used']} tokens > 200), the tool that call requested was **not** executed. final_answer: \"{demo_a['final_answer']}\"\n"
        f"- Timeout demo: `elapsed >= time_limit_seconds` is true at the top of the first iteration (0 >= 0), so the run stops before any model call. final_answer: \"{demo_b['final_answer']}\"")
with open("outputs/agent_demos.jsonl", "a", encoding="utf-8") as f:
    for name, d in [("S1_token_limit_demo", demo_a), ("S1_timeout_demo", demo_b)]:
        f.write(json.dumps({"scenario_id": "S1", "run_id": name, "status": d["status"], "final_answer": d["final_answer"],
                            "llm_calls": d["llm_calls"], "tokens_used": d["tokens_used"], "steps": d["steps"]}) + "\n")

# ------------------------------------------------------------------ crash and resume
rep.add("### D.8 Resumable run (crash after the second tool, then resume)")
RID = "S1_crash_resume_demo"
crashed = False
try:
    run_agent(s1_msg, "S1", run_id=RID, crash_after_tools=2)
except SimulatedCrash as e:
    crashed = True
    crash_msg = str(e)
assert crashed
ck = json.load(open(f"state/{RID}.json"))
rep.add(f"Simulated crash: *{crash_msg}*. Checkpoint `state/{RID}.json` holds run_id, scenario_id, the full message list ({len(ck['messages'])} messages), tokens_used ({ck['tokens_used']}), "
        f"step ({ck['step']}) and tools already executed ({', '.join(ck['tools_executed'])}).")
res = resume_agent(RID)
final_ck = json.load(open(f"state/{RID}.json"))
tools = final_ck["tools_executed"]
assert tools == ["get_order", "search_policy", "calculate_refund", "issue_refund"], tools
assert res["status"] == "REFUND" and abs(refund_amount(res) - 128.25) <= 0.01
with open("outputs/agent_demos.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps({"scenario_id": "S1", "run_id": RID, "status": res["status"], "final_answer": res["final_answer"],
                        "llm_calls": res["llm_calls"], "tokens_used": res["tokens_used"], "steps": res["steps"]}) + "\n")
rep.add(f"`resume_agent('{RID}')` loaded the checkpoint and continued: final status **{res['status']}**, refund INR {refund_amount(res):,.2f}, {res['llm_calls']} model calls in total.\n\n"
        "**Tools executed across the crash and the resume, in order (4 executions, none repeated):** " + " -> ".join(f"`{t}`" for t in tools) + ".")

# ------------------------------------------------------------------ ReAct vs plan-and-execute
rep.add("### D.9 ReAct versus plan-and-execute")
t1, t5, t6, t4 = (results[s]["tokens_used"] for s in ["S1", "S5", "S6", "S4"])
text = (f"Measured with the mock model, my ReAct agent needs 5 model calls for S1 ({t1:,} estimated tokens), 4 for S4, 3 for S5 ({t5:,} tokens) and 2 for S6 ({t6:,} tokens). "
        "Every call re-sends the whole growing message history, so tokens rise faster than calls. A plan-and-execute design asks the model once for the whole step list "
        "(get_order, search_policy, calculate_refund, issue_refund) plus at most one call to write the answer, so it needs about 2 calls and clearly fewer tokens, and the model is not "
        "re-invoked after every tool.\n\n"
        "Its weakness is the unexpected observation. In S6 (NOT_FOUND) and S5 (a cancelled order) the plan made before seeing the data is wrong: the remaining steps must be skipped or the "
        "plan revised, and a blindly executed tail could calculate or even issue a refund for an order that is not refundable. The ReAct agent reacts immediately: it stops after 2 calls for "
        "NOT_FOUND and after 3 calls for a cancelled order, and routes a refund above INR 1000 to a human only after seeing the computed amount.\n\n"
        "For this refund workflow I would choose ReAct (with the hard limits of D.7): the chains are short (at most 5 calls), money-moving steps depend on earlier observations, "
        "and the extra tokens are cheap compared with one wrong refund. Plan-and-execute becomes attractive only for long, predictable workflows.")
n_words = len(text.split())
assert n_words <= 250, n_words
rep.add(text)
rep.add(f"_(word count: {n_words})_")
rep.add(f"Outputs: `outputs/agent_runs.jsonl` ({len(sc)} lines, one per scenario run), `outputs/agent_demos.jsonl` (the three demonstration runs), "
        f"`outputs/refund_ledger.jsonl` ({sum(1 for _ in open('outputs/refund_ledger.jsonl'))} issued/escalated actions, each tagged with its run_id; the crash-and-resume demo adds one `issue_refund` for S1) and `state/` ({len(glob.glob('state/*.json'))} checkpoint files).")
rep.save()
