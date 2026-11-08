"""Eval on held-out SciQ validation split, clean examples only (no corruption).
Reports plain 4-way accuracy — the no-uncertainty baseline.
Usage: python src/jev_uncertain/eval_clean.py --config configs/train_direct.yaml
"""
import argparse, sys, random
from pathlib import Path
import torch
import yaml
sys.path.insert(0, "src")
from jev_uncertain.dataset import load_sciq
from jev_uncertain.train_direct import prompt_for, LETTERS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_direct.yaml")
    ap.add_argument("--data_config", default="configs/data.yaml")
    ap.add_argument("--ckpt", default=None)  # outputs/jev_direct/lora ; None = base model
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config)); dcfg = yaml.safe_load(open(args.data_config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = cfg.get("mode", "head")
    rng = random.Random(1234)
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    mn = cfg.get("model_name", "Qwen/Qwen3-0.6B")
    tok = AutoTokenizer.from_pretrained(mn, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    backbone = AutoModelForCausalLM.from_pretrained(
        mn, trust_remote_code=True,
        torch_dtype=(torch.float16 if device.type == "cuda" else torch.float32))
    head = None
    if args.ckpt:
        from peft import PeftModel
        backbone = PeftModel.from_pretrained(backbone, Path(args.ckpt) / "lora" if (Path(str(args.ckpt)) / "lora").exists() else args.ckpt)
        hp = Path(str(args.ckpt)) / "head.pt"
        if mode == "head" and hp.exists():
            head = torch.nn.Linear(backbone.config.hidden_size, 5)
            head.load_state_dict(torch.load(hp, map_location="cpu"))
    if mode == "head" and head is None and args.ckpt is None:
        print("[eval] NOTE: no ckpt + head mode with random head — use --ckpt or run gen-equivalent argmax over letter logits")
    backbone.to(device); backbone.eval()
    if head: head.to(device); head.eval()

    ds = load_dataset(dcfg.get("dataset_name", "allenai/sciq"), split="validation")
    rows = [r for r in ds if (r.get("support") or "").strip()]
    print(f"[eval] clean validation rows with support: {len(rows)}")
    correct = total = 0
    wrong_ids = []
    import torch.nn.functional as F
    with torch.no_grad():
        for k, r in enumerate(rows):
            gold = r["correct_answer"]; distract = [r["distractor1"], r["distractor2"], r["distractor3"]]
            opts = rng.sample([gold] + distract, 4)
            gi = opts.index(gold)
            unc = dcfg.get("uncertain_text", "I am uncertain.")
            p = prompt_for(r.get("support", ""), r["question"], opts + [unc])
            enc = tok(p, return_tensors="pt", truncation=True, max_length=cfg.get("max_len", 512))
            enc = {kk: vv.to(device) for kk, vv in enc.items()}
            if mode == "head" and head is not None:
                h = backbone(**enc, output_hidden_states=True).hidden_states[-1][:, -1, :]
                logits = head(h.float())[0, :4]  # ignore uncertain slot
                pr = int(logits.argmax())
            else:  # letter-logit baseline works with or without ckpt
                nxt = backbone(**enc).logits[0, -1]
                vocab = tok.get_vocab()
                ids = [(tok.convert_tokens_to_ids(" " + L) if (" " + L) in vocab else tok.convert_tokens_to_ids(L)) for L in LETTERS[:4]]
                tids = torch.tensor(ids, device=device)
                pr = int(tids[nxt[tids].argmax()])
            total += 1; correct += (pr == gi)
            if pr != gi and len(wrong_ids) < 10: wrong_ids.append((k, gi, pr))
    print(f"[eval-clean] acc={correct/max(1,total):.3f} ({correct}/{total}) mode={mode} ckpt={args.ckpt}")
    for w in wrong_ids: print("  miss:", w)

if __name__ == "__main__":
    main()
