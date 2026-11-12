# CredIntel
A domain-specialized LLM for US credit repair, consumer protection laws, and regulatory compliance. Exploring continued pretraining, LoRA-based fine-tuning, and preference optimization to develop reliable legal knowledge and reasoning capabilities.

Milestone 1 asks: does a LoRA adapter trained on ~400 verified examples make Qwen3.5-4B more accurate on US credit repair than the base model, with and without BM25 retrieval? Runs locally on an M3 Pro (18 GB) with mlx-lm.

## Locked test set

| File | Items | SHA-256 |
|---|---|---|
| `data/evaluation/test.jsonl` | 200 | `c5dd8c4a436e22b9b8cc1abb8249dc71c7005fb9d879692fb2762b1a5e98674b` |
| `data/evaluation/validation.jsonl` | 50 | (not locked; used freely while tuning) |

Locked 2026-11-12. Never open `test.jsonl` while writing SFT data; it is run once per arm at the end.
Check: `shasum -a 256 data/evaluation/test.jsonl`

## Pipeline

```bash
.venv/bin/python scripts/convert_to_md.py      # sources -> data/md (Apple Vision OCR for scans)
.venv/bin/python scripts/clean_corpus.py       # -> data/clean
.venv/bin/python scripts/chunk_corpus.py       # -> data/processed/domain_corpus/chunks.jsonl
.venv/bin/python scripts/generate_items.py     # benchmark candidates via gpt-5.5
.venv/bin/python scripts/verify_items.py check # auto-checks -> review.csv; then `lock`
.venv/bin/python scripts/retrieve.py "query"   # BM25 over chunks
.venv/bin/python scripts/evaluate_model.py --arm B0|B1|S1|S2 [--split validation|test]
```

## Decisions

| Date | Decision |
|---|---|
| 2026-11-11 | Model `mlx-community/Qwen3.5-4B-MLX-4bit`; loads text-only in mlx-lm 0.32.2 |
| 2026-11-11 | Thinking **off** for every run (`enable_thinking=False`) |
| 2026-11-11 | OCR with Apple Vision (`ocrmac`), not Tesseract: handles curved phone photos, red underlining, show-through |
| 2026-11-11 | Benchmark generator: OpenAI `gpt-5.5`. Judge must be a different model family (TBD) |
| 2026-11-12 | 12 CFR 1022.38 excluded: CFPB 2025 medical-debt rule vacated (E.D. Tex., 2025-07-11). 1022.30 text not yet checked |
| 2026-11-12 | Credit-report screenshot pages dropped/trimmed (`clean_corpus.EXCLUDE_PAGES`) |
| 2026-11-12 | Not sent to external AI / not retrieved: `_OceanofPDF.com_` books (rights unresolved), `loandoc*`, `identityiq*` |
| 2026-11-12 | Benchmark excludes coaching-book chapters and duplicate Account Types files; MC options shuffled at lock |

## Benchmark build log

- 798 candidates generated (gpt-5.5) → 673 passed auto-checks → 80 excluded after an external review → 250 kept (`keep=y`).
- Review: candidate review was done by an AI reviewer, not a human expert. Treat as a limitation; plan a 30-50 item expert spot-check.
- Split: validation 50 / test 200, stratified by category.

| Category | Validation | Test |
|---|---|---|
| Legal knowledge | 12 | 48 |
| Legal application | 10 | 40 |
| Report reading | 8 | 32 |
| Error detection | 6 | 24 |
| Evidence assessment | 5 | 20 |
| Abstention | 5 | 20 |
| False premise | 4 | 16 |

## Phase 1 smoke test (2026-11-11)

| Measure | Value |
|---|---|
| Inference | ~48 tok/s, 2.6 GB peak |
| 20-step LoRA (batch 1, seq 1024, 16 layers) | val loss 2.93 → 1.20 |
| Batch 2, seq 2048 | 6.8 GB peak (short smoke rows) |

## Run log

| Run | Config | Train loss | Val loss | Peak mem | Wall time | Val score |
|---|---|---|---|---|---|---|
| | | | | | | |
