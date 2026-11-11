"""Build 50 throwaway chat rows from the Metro-2 scenario CSVs for the Phase 1 smoke test."""
import csv, glob, json, os, random

rows = []
for f in sorted(glob.glob("credit repair llm data/Metro-2_*_Scenarios.csv")):
    field = os.path.basename(f)[8:].split("_Scenarios")[0].replace("_", " ")
    with open(f, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r.get("Scenario") and r.get("Correction"):
                q = f"Metro-2 {field}: what does '{r['Scenario']}' mean and how is it corrected?"
                a = f"{r['Description']} Impact: {r['Impact on Credit Report']} Correction: {r['Correction']}"
                rows.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]})
random.Random(0).shuffle(rows)
assert len(rows) >= 50, len(rows)
os.makedirs("data/smoke", exist_ok=True)
for name, part in [("train", rows[:40]), ("valid", rows[40:50])]:
    with open(f"data/smoke/{name}.jsonl", "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in part)
print(len(rows), "rows available; wrote 40 train / 10 valid")
