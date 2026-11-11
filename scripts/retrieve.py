"""BM25 retrieval over the corpus chunks (Phase 4, arm B1/S2).

Usage: .venv/bin/python scripts/retrieve.py "what must a CRA do after a dispute" [-k 5]
Import: from retrieve import Retriever; Retriever().search(q, k=5) -> list of chunk dicts
"""
import argparse, json, re
from rank_bm25 import BM25Okapi

CHUNKS = "data/processed/domain_corpus/chunks.jsonl"
# Not used for retrieval: no rights decision yet / may contain real personal data (same as generate_items.py)
EXCLUDE = re.compile(r"oceanofpdf|loandoc|identityiq", re.I)
STOP = set("a an the of to and or in on for is are be by with as at that this it from what which who does do".split())


def tokenize(text):
    # keep section ids like 1681i, 1692g, 1022.123, 605a whole; drop § and punctuation
    toks = re.findall(r"\d+[a-z]?(?:\.\d+)?|[a-z]+", text.lower())  # "1681i(a)" -> "1681i", "a"
    return [t for t in toks if t not in STOP]


class Retriever:
    def __init__(self, path=CHUNKS):
        self.chunks = [c for c in map(json.loads, open(path)) if not EXCLUDE.search(c["source"])]
        self.bm25 = BM25Okapi([tokenize(c["text"] + " " + " ".join(c["section_path"])) for c in self.chunks])

    def search(self, query, k=5):
        scores = self.bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [self.chunks[i] for i in top if scores[i] > 0]


def format_context(chunks):
    """Each passage labelled with source and page so the model can cite it (plan, Phase 4)."""
    return "\n\n".join(f"[{i + 1}] {c['source']}, p. {c['page']}"
                       + (f" ({' > '.join(c['section_path'])})" if c["section_path"] else "")
                       + f"\n{c['text']}" for i, c in enumerate(chunks))


def _selftest():
    assert "1681i" in tokenize("Under § 1681i(a), the agency") and "1006.34" in tokenize("see 12 CFR 1006.34")


if __name__ == "__main__":
    _selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    a = ap.parse_args()
    for c in Retriever().search(a.query, a.k):
        print(f"{c['chunk_id']}  ({c['source']}, p{c['page']})\n   {' '.join(c['text'].split())[:160]}\n")
