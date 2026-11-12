"""Verify benchmark candidates (Phase 3) and lock the test set.

  check: automatic checks -> data/evaluation/review.csv for your manual review
         (quote appears word for word in the chunk, MC shape, banned phrases)
  lock:  reads review.csv (keep column = y), splits 50 validation / rest test (stratified by category),
         writes validation.jsonl + test.jsonl and prints the test SHA-256 for the README.
Usage: .venv/bin/python scripts/verify_items.py check|lock
"""
import csv, hashlib, json, os, random, re, sys, unicodedata
from collections import defaultdict

CHUNKS = "data/processed/domain_corpus/chunks.jsonl"
CAND = "data/evaluation/candidates.jsonl"
REVIEW = "data/evaluation/review.csv"
BANNED = re.compile(r"guarantee|will be removed|will be deleted|definitely illegal|always win", re.I)
N_VALID = 50


def norm(s):  # word-for-word, but ignore whitespace, curly-vs-straight quotes, dashes and case
    s = unicodedata.normalize("NFKC", s).translate(str.maketrans("‘’“”–—", "''\"\"--"))
    return re.sub(r"\s+", " ", s).strip().lower()


def problems(item, chunk):
    p = []
    if chunk is None:
        p.append("chunk_missing")
    elif norm(item["support_quote"]) not in norm(chunk["text"]):
        p.append("quote_not_in_chunk")
    if len(item["support_quote"].split()) < 5:
        p.append("quote_too_short")
    if item["format"] == "mc":
        if len(item["options"]) != 4 or not re.match(r"^\(?[A-D]\b", item["reference_answer"].strip()):
            p.append("bad_mc_shape")
    if BANNED.search(item["reference_answer"]):
        p.append("banned_phrase")
    if len(item["key_points"]) < 2:
        p.append("few_key_points")
    return p


def check():
    chunks = {c["chunk_id"]: c for c in map(json.loads, open(CHUNKS))}
    items = list(map(json.loads, open(CAND)))
    fails = defaultdict(int)
    # keep earlier review marks/notes when re-run after adding candidates
    prev = {r["row"]: r for r in csv.DictReader(open(REVIEW))} if os.path.exists(REVIEW) else {}
    with open(REVIEW, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["keep", "row", "category", "format", "question", "options", "reference_answer", "support_quote", "source", "page", "auto_problems"])
        for i, it in enumerate(items):
            p = problems(it, chunks.get(it["chunk_id"]))
            for x in p:
                fails[x] += 1
            if p:  # auto-fail = discard (plan); not shown for review
                continue
            old = prev.get(str(i), {})
            w.writerow([old.get("keep", ""), i, it["category"], it["format"], it["question"], " | ".join(it["options"]),
                        it["reference_answer"], it["support_quote"], it["source"], it["page"], old.get("auto_problems", "")])
    passed = len(items) - sum(1 for it in items if problems(it, chunks.get(it["chunk_id"])))
    print(f"{len(items)} candidates, {passed} passed auto-checks -> {REVIEW}")
    print("auto-fail reasons:", dict(fails))
    print("Next: open review.csv, put y in 'keep' for items whose quote supports the answer and that have one defensible answer.")


def shuffle_mc(item, rng):
    """Generator put the answer at A/B 74% of the time; shuffle options and relabel the answer letter."""
    if item["format"] != "mc":
        return item
    strip = lambda s: re.sub(r"^\(?[A-D][.)]?\s*[—–-]?\s*", "", s.strip())
    m = re.match(r"^\(?([A-D])\b[.)]?\s*[—–-]?\s*(.*)", item["reference_answer"].strip(), re.S)
    gold = "ABCD".index(m.group(1))
    opts = [strip(o) for o in item["options"]]
    order = list(range(4)); rng.shuffle(order)
    new_gold = "ABCD"[order.index(gold)]
    return dict(item, options=[f"{'ABCD'[i]}. {opts[j]}" for i, j in enumerate(order)],
                reference_answer=f"{new_gold}. {m.group(2)}")


def lock():
    items = list(map(json.loads, open(CAND)))
    srng = random.Random(7)
    kept = [shuffle_mc(items[int(r["row"])], srng) for r in csv.DictReader(open(REVIEW)) if r["keep"].strip().lower() == "y"]
    by_cat = defaultdict(list)
    for it in kept:
        by_cat[it["category"]].append(it)
    rng = random.Random(2026)
    valid, test = [], []
    for cat, xs in sorted(by_cat.items()):
        rng.shuffle(xs)
        k = round(N_VALID * len(xs) / len(kept))  # validation share proportional to category size
        valid += xs[:k]; test += xs[k:]
    for name, xs, prefix in [("validation", valid, "V"), ("test", test, "T")]:
        with open(f"data/evaluation/{name}.jsonl", "w") as f:
            for j, it in enumerate(xs):
                f.write(json.dumps({"id": f"{prefix}-{j + 1:04d}", **it}, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(open("data/evaluation/test.jsonl", "rb").read()).hexdigest()
    print(f"kept {len(kept)}: validation {len(valid)}, test {len(test)}")
    print(f"test.jsonl SHA-256: {digest}  <- record in README; never open test.jsonl while writing SFT data")


def _selftest():
    c = {"text": "The agency shall  conduct a “reasonable” reinvestigation within 30 days."}
    it = {"support_quote": "conduct a \"reasonable\" reinvestigation within 30 days", "format": "short",
          "options": [], "reference_answer": "Within 30 days.", "key_points": ["a", "b"]}
    assert problems(it, c) == [], problems(it, c)
    assert "quote_not_in_chunk" in problems(dict(it, support_quote="within 45 days of receipt"), c)
    assert "banned_phrase" in problems(dict(it, reference_answer="This will be removed."), c)
    mc = {"format": "mc", "options": ["A. right", "B. wrong1", "C. wrong2", "D. wrong3"], "reference_answer": "A — because right."}
    for seed in range(8):
        s = shuffle_mc(mc, random.Random(seed))
        g = s["reference_answer"][0]
        assert s["options"]["ABCD".index(g)] == f"{g}. right" and s["reference_answer"] == f"{g}. because right.", s


if __name__ == "__main__":
    _selftest()
    {"check": check, "lock": lock}[sys.argv[1] if len(sys.argv) > 1 else "check"]()
