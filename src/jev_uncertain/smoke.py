"""Smoke: build 8 rows in-memory (no HF download), tokenize, collate, forward stub."""
import sys, random
sys.path.insert(0, "src")
from jev_uncertain.tokens import add_special_tokens
from jev_uncertain.dataset import build
from jev_uncertain.data_loader import JEVDataset, make_collator

print("[smoke] building fake sciq rows...")
rng = random.Random(0)
fake = [{"question": f"Q{i}?", "correct_answer": f"gold{i}",
         "distractors": [f"w{i}a", f"w{i}b", f"w{i}c"],
         "support": f"Support sentence for {i}. It mentions gold{i}."} for i in range(8)]
ratios = {"clean": .65, "answer_absent_hard": .08, "answer_absent_knn": .07,
          "info_missing": .10, "preserve": .10}
rows, counts = build(fake, ratios, seed=0, uncertain_text="I am uncertain.")
print("[smoke] counts:", counts)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
add_special_tokens(tok)
print("[smoke] vocab:", len(tok))
ds = JEVDataset(rows, tok, max_len=128)
col = make_collator(tok, max_len=128)
batch = col([ds[i] for i in range(8)])
print("[smoke] batch:", batch["input_ids"].shape, "pool:", batch["pool_idx"].shape,
      "omask:", batch["option_mask"].tolist())
print("[smoke] position_ids:", batch["position_ids"])
# forward through random head math only (no backbone download): gather fake hiddens
import torch
from jev_uncertain.model import SetHead
B, H = 8, 32
feats = torch.randn(B, 5, H)
head = SetHead(H)
logits = head(feats, batch["option_mask"])
print("[smoke] logits:", logits.shape, "sample:", logits[0].tolist())
print("[smoke] OK")
