"""Part D tasks 1-3: chunking, TF-IDF retrieval and grounded answering over data/policies/*.txt (offline mock LLM)."""
import glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm

POLICY_DIR = os.path.join("data", "policies")
CHUNK_WORDS = 40


def chunk_words(text, size=CHUNK_WORDS, overlap=10):
    """Split into `size`-word chunks; each next chunk starts `size - overlap` words later.
    Stop after the first chunk whose end reaches or passes the last word."""
    words, step, chunks, start = text.split(), size - overlap, [], 0
    while True:
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
        start += step
    return chunks


def load_chunks(overlap=10, size=CHUNK_WORDS):
    out = []
    for path in sorted(glob.glob(os.path.join(POLICY_DIR, "*.txt"))):
        with open(path, encoding="utf-8") as f:
            for c in chunk_words(f.read(), size, overlap):
                out.append({"source": os.path.basename(path), "text": c})
    return out


class Index:
    def __init__(self, overlap=10):
        self.overlap = overlap
        self.chunks = load_chunks(overlap)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        sims = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        order = sims.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": round(float(sims[i]), 4), "text": self.chunks[i]["text"]} for i in order]

    def answer(self, question):
        hits = self.retrieve(question, k=3)
        context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
        messages = [
            {"role": "system", "content": "You are a POLICY ASSISTANT for Instamart support. Answer only from the supplied context. "
                                           "If the context does not contain the answer, say that you could not find the information."},
            {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"}]
        resp = mock_llm.chat(messages)
        return resp.choices[0].message.content, [h["source"] for h in hits]


_DEFAULT = None


def default_index():
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Index(overlap=10)
    return _DEFAULT


def retrieve(query, k=3):
    return default_index().retrieve(query, k)


def answer(question):
    return default_index().answer(question)
