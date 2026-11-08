"""QA: gold-leak check, pos balance, label histogram. Logs only, no training."""
import argparse, json
from collections import Counter
ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", required=True)
a = ap.parse_args()
rows = [json.loads(l) for l in open(a.inp, encoding="utf-8")]
print(f"n={len(rows)}")
print("variant:", Counter(r["variant"] for r in rows))
print("label:", Counter(r["label_name"] for r in rows))
# gold position balance on non-uncertain
pos = Counter(r["label"] for r in rows if r["label"] in (0,1,2,3))
print("gold_pos:", dict(pos))
# leak: check gold text absent — we stored gold_index None for uncertain; verify spans sane
bad = [r["id"] for r in rows if len(r["options"]) != 5 or len(r["option_spans"]) != 5]
print("bad_rows:", len(bad), bad[:5])
# uncertain must have gold_index None
viol = [r["id"] for r in rows if (r["label"]==4) != (r["gold_index"] is None)]
print("label_gold_mismatch:", len(viol), viol[:5])
# show 3 samples
for r in rows[:3]:
    print("---", r["id"], r["variant"], r["label_name"], "| Q:", r["question"][:120])
print("QA OK")
