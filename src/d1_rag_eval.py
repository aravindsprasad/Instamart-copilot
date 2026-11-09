"""Part D tasks 1-4: chunk counts, retrieval check, grounded answers and evaluation (overlap 10 and overlap 0)."""
import collections
import pandas as pd
from rag import Index
from util import Report, md_table

rep = Report("part_d_rag")
idx = Index(overlap=10)
idx0 = Index(overlap=0)

rep.add("### D.1 Chunking (40 words, overlap 10 words)")
cnt = collections.Counter(c["source"] for c in idx.chunks)
ref = {"refund_policy.txt": 4, "delivery_policy.txt": 3, "escalation_policy.txt": 3, "payment_policy.txt": 3, "privacy_policy.txt": 2}
rep.add(md_table(pd.DataFrame({"document": list(ref), "chunks": [cnt[k] for k in ref], "reference": list(ref.values())})))
rep.add(f"Total chunks: {len(idx.chunks)} (reference 15). Every chunk records its source file name. With overlap 0 there are {len(idx0.chunks)} chunks.")
assert len(idx.chunks) == 15 and all(cnt[k] == v for k, v in ref.items())

rep.add("### D.2 TF-IDF retrieval")
hits = idx.retrieve("refund policy late delivery")
rep.add("Query `refund policy late delivery` -> top three: " + ", ".join(f"{h['source']} ({h['score']:.3f})" for h in hits) +
        " (all three from refund_policy.txt, as required).")
assert all(h["source"] == "refund_policy.txt" for h in hits)
rep.add("**TF and IDF.** Term frequency (TF) measures how often a word occurs inside one chunk, while inverse document frequency (IDF) measures how rare the word is across all chunks, "
        "so words that are frequent in a chunk but rare overall get the highest weights. A limitation is that TF-IDF matches exact words only: two phrasings with the same meaning but "
        "different words (for example \"money back\" versus \"refund\") get no similarity, and a question whose key words are split across a chunk boundary can retrieve the wrong chunk.")

def evaluate(index):
    ev = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    ans = ev[ev.expected_doc != "NONE"]
    unans = ev[ev.expected_doc == "NONE"]
    doc_ok = fact_ok = fb_ok = 0
    detail, failures = [], []
    for r in ans.itertuples():
        text, sources = index.answer(r.question)
        d, f = r.expected_doc in sources, r.expected_fact in text
        doc_ok += d; fact_ok += f
        detail.append((r.question, d, f))
        if not (d and f):
            failures.append((r.question, r.expected_doc, r.expected_fact, sources, text))
    for r in unans.itertuples():
        text, _ = index.answer(r.question)
        fb_ok += text.startswith("I could not find")
    return dict(n=len(ans), doc=doc_ok, fact=fact_ok, nun=len(unans), fb=fb_ok), failures, detail

rep.add("### D.3 Augment and answer")
rep.add("`answer(question)` retrieves three chunks, then sends two messages to `mock_llm.chat`: a system message containing the words POLICY ASSISTANT (answer only from the context, otherwise say the information could not be found) "
        "and a user message `CONTEXT:` + newline + the three chunks each prefixed with `[<source file>]` on its own line + newline `QUESTION:` + newline + the question. It returns the model text and the sources used.")
t, s = idx.answer("How many days do I have to raise a refund request?")
rep.add(f"Example: *How many days do I have to raise a refund request?* -> \"{t}\" (sources: {', '.join(s)}).")

rep.add("### D.4 Retrieval and grounding evaluation (14 questions)")
r10, f10, _ = evaluate(idx)
r0, f0, _ = evaluate(idx0)
rep.add(md_table(pd.DataFrame({
    "measure": ["answerable: expected_doc in top-3 sources", "answerable: answer contains expected_fact", "unanswerable: fallback 'I could not find...'"],
    "overlap 10": [f"{r10['doc']}/{r10['n']}", f"{r10['fact']}/{r10['n']}", f"{r10['fb']}/{r10['nun']}"],
    "overlap 0": [f"{r0['doc']}/{r0['n']}", f"{r0['fact']}/{r0['n']}", f"{r0['fb']}/{r0['nun']}"],
    "reference (overlap 10)": ["12/12", "12/12", "2/2"]})))
assert (r10["doc"], r10["fact"], r10["fb"]) == (12, 12, 2) and (f10 == [])
assert len(f0) >= 1
import re
lines = []
for q, doc, fact, sources, text in f0:
    doc_text = " ".join(open(f"data/policies/{doc}", encoding="utf-8").read().split())
    sent = next(x for x in re.split(r"(?<=[.])\s+", doc_text) if fact in x)
    whole = any(sent in c["text"] for c in idx0.chunks if c["source"] == doc)
    lines.append(f"- *{q}* (expected fact \"{fact}\" in {doc}): the answer was \"{' '.join(text.split())}\". "
                 + ("The sentence that holds the fact is cut by a chunk boundary: no single 40-word chunk contains the whole sentence "
                    f"(\"{sent}\"), so the model only sees a fragment of it and answers with the wrong half."
                    if not whole else "The sentence is intact in a chunk, but the retrieved chunks did not carry it."))
rep.add(f"With zero overlap {len(f0)} answerable question(s) break:\n\n" + "\n".join(lines) +
        "\n\nOverlap keeps the words around a chunk boundary in two neighbouring chunks, so an answer that straddles the boundary still appears whole in at least one chunk.")
rep.save()
