"""Eval a trained SetHead checkpoint (loacl/head.pt) on held-out SciQ validation.
Two suites: (1) clean 4-way baseline (uncertain slot ignored), (2) corrupted
5-way with uncertain (same builder as training). Reports acc + recall_unc.
Usage: python src/jev_uncertain/eval_head.py --ckpt loacl/head.pt
"""
import argparse, sys, random
from pathlib import Path
import torch
import yaml
sys.path.insert(0, "src")
from jev_uncertain.tokens import add_special_tokens
from jev_uncertain.dataset import load_sciq, build
from jev_uncertain.data_loader import JEVDataset, make_collator
from jev_uncertain.model import JEVClassifier

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="local/head.pt")
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--data_config", default="configs/data.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config)); dcfg = yaml.safe_load(open(args.data_config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval-head] ckpt={args.ckpt} device={device}")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    mn = cfg.get("model_name", "Qwen/Qwen3-0.6B")
    tok = AutoTokenizer.from_pretrained(mn, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    add_special_tokens(tok)
    backbone = AutoModelForCausalLM.from_pretrained(
        mn, trust_remote_code=True,
        torch_dtype=(torch.float16 if device.type == "cuda" else torch.float32))
    backbone.resize_token_embeddings(len(tok))
    ckpt = torch.load(args.ckpt, map_location="cpu")
    print("[eval-head] ckpt keys:", list(ckpt.keys()), "T=", ckpt.get("T"))
    if ckpt.get("lora") is not None:
        try:
            from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
            lc = LoraConfig(r=cfg.get("lora_r", 16), lora_alpha=cfg.get("lora_alpha", 32),
                            lora_dropout=cfg.get("lora_dropout", 0.05),
                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                            bias="none", task_type="CAUSAL_LM")
            backbone = get_peft_model(backbone, lc)
            set_peft_model_state_dict(backbone, ckpt["lora"])
            print("[eval-head] LoRA restored")
        except Exception as e:
            print("[eval-head] LoRA restore skipped:", e)
    clf = JEVClassifier(backbone, backbone.config.hidden_size,
                        heads=cfg.get("heads", 4), dropout=0.0)
    clf.set.load_state_dict(ckpt["set"])
    clf.to(device); clf.eval()
    T = float(ckpt.get("T", 1.0))

    from datasets import load_dataset
    rng = random.Random(1234)
    ds = load_dataset(dcfg.get("dataset_name", "allenai/sciq"), split="validation")
    valrows = [r for r in ds if (r.get("support") or "").strip()]
    print(f"[eval-head] validation rows with support: {len(valrows)}")

    from torch.utils.data import DataLoader
    import torch.nn.functional as F
    col = make_collator(tok, max_len=cfg.get("max_len", 512))

    # Suite 1: clean — gold among A-D, argmax over [:4]
    clean = []
    for r in valrows:
        gold = r["correct_answer"]
        opts = rng.sample([gold] + [r["distractor1"], r["distractor2"], r["distractor3"]], 4)
        clean.append({"id": "c", "context": r.get("support", ""), "question": r["question"],
                      "options": opts + [dcfg.get("uncertain_text", "I am uncertain.")],
                      "label": opts.index(gold)})
    dl = DataLoader(JEVDataset(clean, tok, cfg.get("max_len", 512)), batch_size=8, collate_fn=col)
    c = t = 0
    with torch.no_grad():
        for b in dl:
            bb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            lg = clf(**{k: bb[k] for k in ("input_ids", "attention_mask", "pool_idx", "option_mask", "position_ids")})["logits"]
            pr = (lg / T)[:, :4].argmax(-1).cpu()
            c += int((pr == b["labels"]).sum()); t += len(pr)
    print(f"[eval-head] CLEAN 4-way acc={c/max(1,t):.3f} ({c}/{t})")

    # Suite 2: corrupted 5-way via training builder on validation split
    examples = [{"question": r["question"], "correct_answer": r["correct_answer"],
                 "distractors": [r["distractor1"], r["distractor2"], r["distractor3"]],
                 "support": r.get("support", "")} for r in valrows]
    ratios = {k: dcfg.get(k, 0) for k in
              ["clean", "answer_absent_hard", "answer_absent_knn", "info_missing", "preserve"]}
    rows, counts = build(examples, ratios, seed=999,
                         uncertain_text=dcfg.get("uncertain_text", "I am uncertain."))
    print("[eval-head] corrupted mix:", counts)
    dl2 = DataLoader(JEVDataset(rows, tok, cfg.get("max_len", 512)), batch_size=8, collate_fn=col)
    tot = cor = uh = ut = 0
    import collections
    per = collections.Counter()
    with torch.no_grad():
        for b in dl2:
            bb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            lg = clf(**{k: bb[k] for k in ("input_ids", "attention_mask", "pool_idx", "option_mask", "position_ids")})["logits"]
            pr = (lg / T).argmax(-1).cpu()
            tot += len(pr); cor += int((pr == b["labels"]).sum())
            is_unc = b["labels"] == 4
            ut += int(is_unc.sum()); uh += int(((pr == 4) & is_unc).sum())
    print(f"[eval-head] CORRUPT 5-way acc={cor/max(1,tot):.3f} ({cor}/{tot}) recall_unc={uh/max(1,ut):.3f} ({uh}/{ut})")

if __name__ == "__main__":
    main()
