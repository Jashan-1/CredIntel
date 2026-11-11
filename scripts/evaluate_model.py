"""Run one evaluation arm and save outputs (Phases 4 and 7).

Arms: B0 plain | B1 plain + BM25 top-5 | S1 adapter | S2 adapter + BM25 top-5.
Greedy decoding, thinking OFF for every arm (decision 2026-11-11). Same prompts for every arm.
MC items are scored here; short answers get "score": null until the LLM judge step is added.
Writes evaluation_results/<arm>/<date>_<split>.jsonl

Usage: .venv/bin/python scripts/evaluate_model.py --arm B0 [--split validation] [--adapter adapters/sft_best] [--limit N]
"""
import argparse, datetime, json, os, re, sys
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

sys.path.insert(0, os.path.dirname(__file__))
from retrieve import Retriever, format_context

MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"
# Draft prompts: tune on validation in Phase 4, then FREEZE before Phase 6 (plan). Every arm uses the same ones.
SYSTEM = ("You are an assistant for US credit reporting and credit repair questions. "
          "Answer the question directly in the first sentence. Name the specific legal provision when one applies. "
          "Distinguish a possible inaccuracy from a legal violation. Never promise deletions, score increases or outcomes. "
          "If the information given is not enough to answer, say what is missing instead of guessing.")
RETRIEVAL_INSTR = ("Use the numbered source passages below when they are relevant, and cite them by source and page. "
                   "If they do not answer the question, say so.\n\nSOURCES:\n")
MC_INSTR = "\n\nOptions:\n{opts}\n\nAnswer with the letter of the correct option first, then a one-sentence explanation."


def build_prompt(item, retriever):
    q = item["question"]
    if item["format"] == "mc":
        q += MC_INSTR.format(opts="\n".join(item["options"]))
    if retriever:
        q = RETRIEVAL_INSTR + format_context(retriever.search(item["question"], 5)) + "\n\nQUESTION:\n" + q
    return q


def mc_letter(text):
    m = re.search(r"\b([A-D])\b", text.strip()[:40])
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["B0", "B1", "S1", "S2"])
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--adapter", default="adapters/sft_best")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=512)
    a = ap.parse_args()

    items = list(map(json.loads, open(f"data/evaluation/{a.split}.jsonl")))[: a.limit or None]
    model, tok = load(MODEL, adapter_path=a.adapter if a.arm in ("S1", "S2") else None)
    retriever = Retriever() if a.arm in ("B1", "S2") else None
    sampler = make_sampler(temp=0.0)  # greedy

    os.makedirs(f"evaluation_results/{a.arm}", exist_ok=True)
    out_path = f"evaluation_results/{a.arm}/{datetime.date.today()}_{a.split}.jsonl"
    correct = n_mc = 0
    with open(out_path, "w") as out:
        for i, it in enumerate(items):
            user = build_prompt(it, retriever)
            prompt = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                                              add_generation_prompt=True, tokenize=False, enable_thinking=False)
            output = generate(model, tok, prompt=prompt, max_tokens=a.max_tokens, sampler=sampler)
            score = None
            if it["format"] == "mc":
                gold, got = mc_letter(it["reference_answer"]), mc_letter(output)
                score = int(gold is not None and got == gold)
                correct += score; n_mc += 1
            out.write(json.dumps({"id": it["id"], "category": it["category"], "format": it["format"], "prompt": user,
                                  "output": output, "score": score}, ensure_ascii=False) + "\n"); out.flush()
            print(f"[{i + 1}/{len(items)}] {it['id']} {it['category']} {'' if score is None else score}", flush=True)
    print(f"MC accuracy {correct}/{n_mc}" if n_mc else "no MC items", "->", out_path)


if __name__ == "__main__":
    assert mc_letter("B. The agency must...") == "B" and mc_letter("(C) because") == "C" and mc_letter("No idea") is None
    main()
