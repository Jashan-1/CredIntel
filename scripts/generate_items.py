"""Generate benchmark candidates from corpus chunks via an external LLM (Phase 3).

One chunk -> one question. Writes data/evaluation/candidates.jsonl (append, resumable).
Usage: .venv/bin/python scripts/generate_items.py [--per-category-multiplier 2] [--model gpt-5.5]
Needs OPENAI_API_KEY in .env.
"""
import argparse, json, os, random, re
from openai import OpenAI

CHUNKS = "data/processed/domain_corpus/chunks.jsonl"
OUT = "data/evaluation/candidates.jsonl"

# Plan, Phase 3 table. Target counts; we generate ~2x and keep the verified ones.
CATEGORIES = {
    "legal_knowledge": (60, "Which provision covers a situation, or what a specific section requires."),
    "legal_application": (50, "Apply a provision to a short scenario with fictional people and accounts."),
    "report_reading": (40, "Give a short synthetic credit report / tradeline excerpt and ask for a specific fact from it."),
    "error_detection": (30, "Give a synthetic tradeline with one possible inaccuracy (balance, date, duplicate, ownership, status) and ask what is wrong."),
    "evidence_assessment": (25, "Describe a dispute situation and ask what evidence is missing to support or settle it."),
    "abstention": (25, "Ask a question the passage and scenario do NOT give enough information to answer; the correct answer says what is missing instead of guessing."),
    "false_premise": (20, "Ask a question that assumes a legal requirement, deadline or right that does NOT exist in the passage; the correct answer rejects the premise."),
}
# Statutes/regulations ground the legal categories; Metro-2 and guides ground report/data categories.
LEGAL = {"legal_knowledge", "legal_application", "false_premise", "abstention"}
# Not sent to the external service: no rights decision yet / may contain real personal data.
EXCLUDE = re.compile(r"oceanofpdf|loandoc|identityiq", re.I)

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["format", "question", "options", "reference_answer", "key_points", "support_quote", "difficulty"],
    "properties": {
        "format": {"type": "string", "enum": ["mc", "short"]},
        "question": {"type": "string"},
        "options": {"type": "array", "items": {"type": "string"}, "description": "4 options A-D for mc, empty for short"},
        "reference_answer": {"type": "string", "description": "for mc: the correct letter then the explanation"},
        "key_points": {"type": "array", "items": {"type": "string"}},
        "support_quote": {"type": "string"},
        "difficulty": {"type": "integer", "enum": [1, 2, 3]},
    },
}

SYSTEM = """You write evaluation questions for a US credit-repair assistant benchmark.
Rules:
- Ground everything ONLY in the given passage. Do not use outside legal knowledge.
- support_quote must be copied WORD FOR WORD from the passage (one contiguous span, 10-60 words) and must support the reference answer.
- The question must have one defensible answer and must not reveal it.
- Use only fictional people, companies and account numbers.
- key_points: 2-4 short facts a correct answer must contain.
- Never state or imply guaranteed deletion, score increases or outcomes.
- If the passage cannot support a good question of the requested category, still return your best attempt; it will be filtered later."""


def pick_chunks(chunks, rng, mult):
    legal = [c for c in chunks if c["doc_type"] == "statute"]
    data = [c for c in chunks if c["doc_type"] != "statute"]
    jobs = []
    for cat, (n, _) in CATEGORIES.items():
        pool = legal if cat in LEGAL else data
        for c in rng.sample(pool, min(len(pool), n * mult)):
            jobs.append((cat, c))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category-multiplier", type=int, default=2)
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--limit", type=int, default=0, help="stop after N new items (0 = all)")
    args = ap.parse_args()
    for line in open(".env"):
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1); os.environ.setdefault(k, v.strip().strip('"'))
    client = OpenAI()

    chunks = [c for c in map(json.loads, open(CHUNKS)) if not EXCLUDE.search(c["source"]) and len(c["text"].split()) > 60]
    jobs = pick_chunks(chunks, random.Random(1234), args.per_category_multiplier)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = {(r["category"], r["chunk_id"]) for r in map(json.loads, open(OUT))} if os.path.exists(OUT) else set()
    made = 0
    with open(OUT, "a") as out:
        for i, (cat, c) in enumerate(jobs):
            if (cat, c["chunk_id"]) in done:
                continue
            fmt = "mc" if random.Random(c["chunk_id"] + cat).random() < 0.4 else "short"  # ~40% MC (plan)
            user = (f"Category: {cat} - {CATEGORIES[cat][1]}\nFormat: {fmt}\n"
                    f"Source: {c['source']} | section: {' > '.join(c['section_path'])} | page {c['page']}\n\n"
                    f"PASSAGE:\n{c['text']}")
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                    response_format={"type": "json_schema", "json_schema": {"name": "item", "strict": True, "schema": SCHEMA}},
                )
                item = json.loads(resp.choices[0].message.content)
            except Exception as e:  # one failure shouldn't stop a long run; rerun to retry
                print(f"[{i}] {cat} {c['chunk_id']}: ERROR {e}", flush=True); continue
            item.update(category=cat, chunk_id=c["chunk_id"], source=c["source"], page=c["page"], generator=args.model)
            out.write(json.dumps(item, ensure_ascii=False) + "\n"); out.flush()
            made += 1
            print(f"[{i + 1}/{len(jobs)}] {cat} {c['chunk_id']}", flush=True)
            if args.limit and made >= args.limit:
                break
    print(f"wrote {made} new items -> {OUT}")


if __name__ == "__main__":
    main()
