"""Chunk data/clean/*.md into data/processed/domain_corpus/chunks.jsonl (Phase 2, steps 4-5, 8).

Splits at headings, carries the heading path, packs paragraphs into 300-600 token chunks,
records the page each chunk starts on. Prints a per-source token report.
Usage: .venv/bin/python scripts/chunk_corpus.py
"""
import csv, json, os, re, sys
from collections import defaultdict

SRC, OUT = "data/clean", "data/processed/domain_corpus/chunks.jsonl"
MIN_TOK, MAX_TOK = 300, 600
PAGE = re.compile(r"^<!-- page (\d+) -->$")
HEAD = re.compile(r"^(#{1,6})\s+\**(.+?)\**\s*$")
STATUTES = {"fcra-revised", "fair-debt-collection-practices-act", "croa 15 usc", "regulation v 12 cfr", "regulation f 12 cfr"}  # the official statute texts only


def tokens(s):  # shortcut: words*1.3 approximates Qwen tokens; swap for the real tokenizer if chunk sizes matter
    return int(len(s.split()) * 1.3)


def ocr_files(log="data/convert_log.txt"):
    # shortcut: file-level OCR flag from the converter log, not per page
    if not os.path.exists(log):
        return set()
    return {m.group(1) for m in re.finditer(r"^ok   (.+?)  \(\d+ pages, [1-9]\d* OCR\)", open(log).read(), re.M)}


def blocks(text):
    """Yield (section_path, page, paragraph)."""
    path, page = [], 1
    for para in re.split(r"\n\s*\n", text):
        lines = []
        for line in para.splitlines():
            s = line.strip()
            if m := PAGE.match(s):
                page = int(m.group(1)); continue
            if m := HEAD.match(s):
                if lines:
                    yield list(path), page, "\n".join(lines); lines = []
                lvl = len(m.group(1))
                path = path[: lvl - 1] + [""] * max(0, lvl - 1 - len(path)) + [m.group(2).strip()]
                continue
            lines.append(line)
        if lines and "".join(lines).strip():
            yield [p for p in path if p], page, "\n".join(lines)


def split_oversize(para):
    """Split a too-long paragraph at line breaks (keeps table rows whole), then by words."""
    if tokens(para) <= MAX_TOK:
        return [para]
    out, cur = [], []
    for line in para.splitlines():
        while tokens(line) > MAX_TOK:
            w = line.split(); n = int(MAX_TOK / 1.3)
            out.append(" ".join(w[:n])); line = " ".join(w[n:])
        if cur and tokens("\n".join(cur + [line])) > MAX_TOK:
            out.append("\n".join(cur)); cur = []
        cur.append(line)
    return out + (["\n".join(cur)] if cur else [])


def chunk_doc(text):
    """Pack paragraphs into chunks; start a new chunk on a heading change once MIN_TOK is reached."""
    cur, cur_path, cur_page = [], None, None
    for path, page, para in blocks(text):
        size = sum(tokens(p) for p in cur)
        if cur and (size + tokens(para) > MAX_TOK or (path != cur_path and size >= MIN_TOK)):
            yield cur_path, cur_page, "\n\n".join(cur); cur = []
        if not cur:
            cur_path, cur_page = path, page
        for piece in split_oversize(para):
            if cur and sum(tokens(p) for p in cur) + tokens(piece) > MAX_TOK:
                yield cur_path, cur_page, "\n\n".join(cur); cur, cur_path, cur_page = [], path, page
            cur.append(piece)
    if cur:
        yield cur_path, cur_page, "\n\n".join(cur)


def shingles(text, k=5):
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}


def dedup(recs, threshold=0.8):
    """Drop exact and near-duplicate chunks (Jaccard on 5-word shingles). Statutes win, then longer sources.

    shortcut: O(n^2) pairwise compare, fine for a few thousand chunks; switch to MinHash LSH past ~20k.
    """
    order = sorted(recs, key=lambda r: (r["doc_type"] != "statute", r["ocr"], r["chunk_id"]))
    kept, kept_sh, seen, dropped = [], [], set(), []
    for r in order:
        key = re.sub(r"\W+", " ", r["text"].lower()).strip()
        sh = shingles(r["text"])
        dup_of = None
        if key in seen:
            dup_of = "exact"
        else:
            for k, s in zip(kept, kept_sh):
                if len(sh & s) / (len(sh | s) or 1) >= threshold:
                    dup_of = k["chunk_id"]; break
        if dup_of:
            dropped.append((r["chunk_id"], dup_of)); continue
        seen.add(key); kept.append(r); kept_sh.append(sh)
    return sorted(kept, key=lambda r: r["chunk_id"]), dropped


def source_date(original, dates):
    # inventory "file" may be the full file name or a prefix of the md name (law files carry a date suffix)
    return next((d for f, d in dates.items() if original == f or original.startswith(f)), None)


def main():
    ocr = ocr_files()
    inv = "sources_inventory.csv"
    dates = {r["file"]: r["source_date"] for r in csv.DictReader(open(inv)) if r["source_date"]} if os.path.exists(inv) else {}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    report = defaultdict(lambda: [0, 0])
    recs = []
    for name in sorted(os.listdir(SRC)):
        if not name.endswith(".md"):
            continue
        original = name[:-3]
        slug = re.sub(r"[^a-z0-9]+", "_", original.lower()).strip("_")[:50]
        doc_type = "statute" if any(k in original.lower() for k in STATUTES) else "book_or_guide"
        for i, (path, page, txt) in enumerate(chunk_doc(open(os.path.join(SRC, name)).read())):
            recs.append({"chunk_id": f"{slug}_{i:04d}", "text": txt, "source": original, "doc_type": doc_type,
                         "section_path": path, "page": page, "source_date": source_date(original, dates),
                         "jurisdiction": "US-federal", "ocr": original in ocr})
    recs, dropped = dedup(recs)
    with open(OUT, "w") as out:
        for rec in recs:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            report[rec["source"]][0] += 1; report[rec["source"]][1] += tokens(rec["text"])
    with open(OUT.replace("chunks.jsonl", "dropped_duplicates.tsv"), "w") as f:
        f.writelines(f"{a}\t{b}\n" for a, b in dropped)
    print(f"dedup dropped {len(dropped)} chunks (see dropped_duplicates.tsv)")
    for src, (n, t) in sorted(report.items(), key=lambda x: -x[1][1]):
        print(f"{t:>8,} tok {n:>5} chunks  {src}")
    print(f"TOTAL {sum(t for _, t in report.values()):,} tokens, {sum(n for n, _ in report.values()):,} chunks -> {OUT}")


def _selftest():
    doc = "<!-- page 1 -->\n# Law\n## 1681i\n" + "\n\n".join(["word " * 100] * 8) + "\n<!-- page 2 -->\n## 1681s\nshort."
    ch = list(chunk_doc(doc))
    assert all(tokens(t) <= MAX_TOK for _, _, t in ch), [tokens(t) for _, _, t in ch]
    assert ch[0][0] == ["Law", "1681i"] and ch[-1][0] == ["Law", "1681s"] and ch[-1][1] == 2, ch[-1][:2]
    base = {"source": "", "section_path": [], "page": 1, "ocr": False}
    t = "The consumer reporting agency shall conduct a reasonable reinvestigation within thirty days " * 3
    recs = [dict(base, chunk_id="book_1", doc_type="book_or_guide", text=t + "extra"),
            dict(base, chunk_id="fcra_1", doc_type="statute", text=t),
            dict(base, chunk_id="other", doc_type="book_or_guide", text="Totally different text about medical debt rules.")]
    kept, dropped = dedup(recs)
    assert [r["chunk_id"] for r in kept] == ["fcra_1", "other"], (kept, dropped)


if __name__ == "__main__":
    _selftest()
    main()
