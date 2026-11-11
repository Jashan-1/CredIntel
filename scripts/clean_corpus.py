"""Clean data/md/*.md into data/clean/ (Phase 2, step 3).

Drops repeating headers/footers and bare page numbers, rejoins hyphenated line breaks,
normalizes ligatures and whitespace, fixes common Tesseract slips.
Keeps `<!-- page N -->` markers (needed for citations), footnotes, § and citations.
Usage: .venv/bin/python scripts/clean_corpus.py [IN_DIR] [OUT_DIR]
"""
import os, re, sys, unicodedata
from collections import Counter

PAGE = re.compile(r"^<!-- page \d+ -->$", re.M)
OCR_FIXES = [  # shortcut: only slips seen in sample pages; extend after the Phase 2 20-chunk check
    (re.compile(r"\bJf\b"), "If"),
    (re.compile(r"\b(it|that|This|this|what|What) 1s\b"), r"\1 is"),
    (re.compile(r"(^|[.!?]\s+)\| (?=[a-z])", re.M), r"\1I "),
    (re.compile(r"\bdon t\b"), "don't"),
    (re.compile(r"\bl (am|have|will|was)\b"), r"I \1"),  # Apple Vision reads "I" as "l"
    (re.compile(r"\byou re\b"), "you're"),
]


# Pages dropped after manual review (2026-11-12): credit-report screenshots that OCR reads column by
# column, so values detach from their labels/bureaus; also partly masked real account numbers.
EXCLUDE_PAGES = {
    "Chapter 9 (1).pdf.md": {5, 6, 8, 9, 11, 12, 13, 14},  # pure screenshots, 0-1 prose lines
}
# Mixed pages (prose + embedded report screenshot): keep prose, drop short table-cell lines.
REPORT_LABELS = ("Account Type", "Date Opened", "High Credit", "Past Due:", "Payment Status", "Two-Year", "Credit Limit", "Bureau Code")


def strip_screenshot_cells(page):
    if sum(k in page for k in REPORT_LABELS) < 4:
        return page
    return "\n".join(l for l in page.splitlines() if len(l.split()) >= 4 or not l.strip())


def norm_key(line):
    return re.sub(r"\d+", "#", line.strip().lower())


def clean(text, drop_pages=frozenset()):
    text = unicodedata.normalize("NFKC", text).replace("­", "")
    text = text.replace("\r\n", "\n").replace("\t", " ")
    pages = PAGE.split(text)
    markers = PAGE.findall(text)
    for i, m in enumerate(markers):  # pages[i + 1] follows markers[i]
        if int(re.search(r"\d+", m).group()) in drop_pages:
            pages[i + 1] = ""
    pages = [strip_screenshot_cells(p) for p in pages]

    def edges(p):  # header/footer zone: first and last 3 non-blank lines, never table rows
        ls = [l for l in p.splitlines() if l.strip() and not l.lstrip().startswith("|")]
        return {norm_key(l) for l in ls[:3] + ls[-3:]}

    # line in the edge zone of most pages = header/footer (only meaningful with enough pages)
    counts = Counter(k for p in pages for k in edges(p))
    repeated = {k for k, n in counts.items() if len(pages) >= 5 and n >= 0.5 * len(pages) and len(k) < 80}
    kept = []
    for p in pages:
        letters = [ch for ch in p if ch.isalpha()]
        if letters and sum(not ("a" <= ch.lower() <= "z") for ch in letters) > 0.05 * len(letters):
            kept.append(""); continue  # mirrored show-through OCR'd as Cyrillic/Vietnamese junk (Ch 1 p4)
        zone = edges(p) & repeated
        kept.append("\n".join(
            l.rstrip() for l in p.splitlines()
            if not (norm_key(l) in zone or re.fullmatch(r"(page\s*)?\d{1,4}", l.strip(), re.I))))
    text = kept[0] + "".join(f"\n{m}\n{p}" for m, p in zip(markers, kept[1:]))
    text = re.sub(r"([A-Za-z])-\n\s*([a-z])", r"\1\2", text)  # hyphenated line break
    for pat, rep in OCR_FIXES:
        text = pat.sub(rep, text)
    text = re.sub(r"[  ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def _selftest():
    pages = "".join(f"<!-- page {i} -->\nThe Perfect Letter\nBody {'abcdef'[i-1] * 3}.\n{i}\n" for i in range(1, 7))
    c = clean(pages)
    assert "The Perfect Letter" not in c and "Body ccc." in c, c
    assert "<!-- page 3 -->" in c and "\n3\n" not in c, c
    assert clean("A credit-\nworthy man. Jf so, it 1s fine.") == "A creditworthy man. If so, it is fine.\n"
    assert "§ 1681i(a)" in clean("See § 1681i(a).")
    junk = clean("<!-- page 1 -->\nGood text here.\n<!-- page 2 -->\nВід возив Бол родо скай мрет\n")
    assert "Good text" in junk and "Від" not in junk and "<!-- page 2 -->" in junk, junk
    d = clean("<!-- page 1 -->\nKeep me.\n<!-- page 2 -->\nDrop me.\n<!-- page 3 -->\nKeep too.\n", {2})
    assert "Keep me." in d and "Drop me." not in d and "Keep too." in d, d
    shot = "Account Type\nDate Opened\nHigh Credit\nPast Due:\n$4,133.00\n1. Comments: Medical (Equifax).\nThese are the tell signs you look for."
    s = strip_screenshot_cells(shot)
    assert "$4,133.00" not in s and "Medical (Equifax)" in s and "tell signs" in s, s


if __name__ == "__main__":
    _selftest()
    src = sys.argv[1] if len(sys.argv) > 1 else "data/md"
    dst = sys.argv[2] if len(sys.argv) > 2 else "data/clean"
    os.makedirs(dst, exist_ok=True)
    before = after = 0
    for name in sorted(os.listdir(src)):
        if not name.endswith(".md"):
            continue
        raw = open(os.path.join(src, name), encoding="utf-8").read()
        out = clean(raw, EXCLUDE_PAGES.get(name, frozenset()))
        open(os.path.join(dst, name), "w", encoding="utf-8").write(out)
        before, after = before + len(raw), after + len(out)
    print(f"cleaned {len(os.listdir(dst))} files: {before:,} -> {after:,} chars")
