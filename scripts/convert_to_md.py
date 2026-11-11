"""Convert every source file to Markdown in data/md/. Scanned PDF pages are OCR'd with Apple Vision (ocrmac).

Usage: .venv/bin/python scripts/convert_to_md.py [SRC_DIR] [OUT_DIR]
Skips files whose .md already exists, so it can be re-run after adding sources.
"""
import os, shutil, sys, tempfile
import pymupdf, pymupdf4llm
from ocrmac import ocrmac
from markitdown import MarkItDown

SRC = sys.argv[1] if len(sys.argv) > 1 else "credit repair llm data"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/md"
MIN_CHARS = 100  # a page with less text than this is treated as scanned (plan, Phase 2)


def ocr_page(page):
    # Apple Vision (macOS, local): handles curved phone photos, red underlining and show-through
    # that broke Tesseract (CHAPTER 4 (1) p2 went from fragments to clean text).
    with tempfile.TemporaryDirectory() as d:
        img = os.path.join(d, "p.png")
        page.get_pixmap(dpi=200, alpha=False).save(img)
        res = ocrmac.OCR(img, recognition_level="accurate", language_preference=["en-US"]).recognize()
        return "\n".join(text for text, _conf, _box in res)


def pdf_to_md(path):
    doc = pymupdf.open(path)
    parts, ocr_pages = [], 0
    for i, page in enumerate(doc):
        if len(page.get_text().strip()) >= MIN_CHARS:
            text = pymupdf4llm.to_markdown(doc, pages=[i], show_progress=False)
        else:
            text, ocr_pages = ocr_page(page), ocr_pages + 1
        parts.append(f"<!-- page {i + 1} -->\n{text.strip()}")
    return "\n\n".join(parts), f"{doc.page_count} pages, {ocr_pages} OCR"


def main():
    os.makedirs(OUT, exist_ok=True)
    md = MarkItDown()
    skipped = []
    for name in sorted(os.listdir(SRC)):
        src, ext = os.path.join(SRC, name), os.path.splitext(name)[1].lower()
        dst = os.path.join(OUT, name + ".md")  # keep ext: X.pdf and X.docx both exist
        if os.path.exists(dst) or name.startswith("."):
            continue
        try:
            if ext == ".md":
                shutil.copyfile(src, dst); note = "copied"
            elif ext == ".pdf":
                body, note = pdf_to_md(src)
                open(dst, "w").write(f"# {name}\n\n{body}\n")
            elif ext in (".docx", ".csv"):
                open(dst, "w").write(md.convert(src).text_content); note = ext
            else:  # .json data / embeddings, .pages: not prose
                skipped.append(name); continue
            print(f"ok   {name}  ({note})", flush=True)
        except Exception as e:  # keep going; one bad file shouldn't stop the batch
            skipped.append(f"{name}: {e}")
            if os.path.exists(dst):
                os.remove(dst)
    print("\nSkipped:", *skipped, sep="\n  ")


if __name__ == "__main__":
    main()
