"""Direct Qwen finetune: mode=head (pooled+Linear5+CE) or mode=gen (emit A-E letter).
LoRA backbone, device-agnostic. No special tokens; plain prompt."""
import argparse, json, sys, random, time
from pathlib import Path
import torch
import yaml
sys.path.insert(0, "src")
from jev_uncertain.dataset import load_sciq, build

LETTERS = ["A", "B", "C", "D", "E"]

def prompt_for(ctx, q, options):
    lines = [f"Context: {ctx}", f"Question: {q}", "Options:"]
    for i, o in enumerate(options):
        lines.append(f"{LETTERS[i]}) {o}")
    lines.append("Answer:")
    return "\n".join(lines)

def eval_clean_heldout(backbone, head, tok, dcfg, cfg, device, mode, seed=1234, cap=None):
    """Held-out SciQ validation, clean 4-way (uncertain slot ignored). Returns acc."""
    rng = random.Random(seed)
    from datasets import load_dataset
    ds = load_dataset(dcfg.get("dataset_name", "allenai/sciq"), split="validation")
    rows = [r for r in ds if (r.get("support") or "").strip()]
    if cap: rows = rng.sample(rows, min(cap, len(rows)))
    correct = total = 0
    was_training = backbone.training
    backbone.eval()
    with torch.no_grad():
        for r in rows:
            gold = r["correct_answer"]
            opts = rng.sample([gold] + [r["distractor1"], r["distractor2"], r["distractor3"]], 4)
            gi = opts.index(gold)
            p = prompt_for(r.get("support", ""), r["question"],
                           opts + [dcfg.get("uncertain_text", "I am uncertain.")])
            enc = tok(p, return_tensors="pt", truncation=True, max_length=cfg.get("max_len", 512))
            enc = {k: v.to(device) for k, v in enc.items()}
            if mode == "head" and head is not None:
                h = backbone(**enc, output_hidden_states=True).hidden_states[-1][:, -1, :]
                pr = int(head(h.float())[0, :4].argmax())
            else:
                nxt = backbone(**enc).logits[0, -1]
                vocab = tok.get_vocab()
                ids = [(tok.convert_tokens_to_ids(" " + L) if (" " + L) in vocab else tok.convert_tokens_to_ids(L)) for L in LETTERS[:4]]
                tids = torch.tensor(ids, device=device)
                pr = int(tids[nxt[tids].argmax()])
            total += 1; correct += (pr == gi)
    if was_training: backbone.train()
    return correct / max(1, total), correct, total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_direct.yaml")
    ap.add_argument("--data_config", default="configs/data.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config)); dcfg = yaml.safe_load(open(args.data_config))
    torch.manual_seed(cfg.get("seed", 42)); random.seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = cfg.get("mode", "head")
    out = Path(cfg.get("out_dir", "outputs/jev_direct")); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "w")
    def log(*a):
        print(*a); print(*a, file=logf, flush=True)
    log(f"[direct] mode={mode} device={device}")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    mn = cfg.get("model_name", "Qwen/Qwen3-0.6B")
    tok = AutoTokenizer.from_pretrained(mn, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    backbone = AutoModelForCausalLM.from_pretrained(
        mn, trust_remote_code=True,
        torch_dtype=(torch.float16 if device.type == "cuda" else torch.float32))
    from peft import LoraConfig, get_peft_model
    lc = LoraConfig(r=cfg.get("lora_r", 16), lora_alpha=cfg.get("lora_alpha", 32),
                    lora_dropout=cfg.get("lora_dropout", 0.05),
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    bias="none", task_type="CAUSAL_LM")
    backbone = get_peft_model(backbone, lc)
    backbone.print_trainable_parameters()
    head = torch.nn.Linear(backbone.config.hidden_size, 5).to(device) if mode == "head" else None
    backbone.to(device); backbone.train()

    ratios = {k: dcfg.get(k, 0) for k in
              ["clean", "answer_absent_hard", "answer_absent_knn", "info_missing", "preserve"]}
    ex = load_sciq(n=dcfg.get("n_train", dcfg.get("n_dry_run", 200)),
                   seed=cfg.get("seed", 42),
                   dataset_name=dcfg.get("dataset_name", "allenai/sciq"))
    rows, counts = build(ex, ratios, seed=cfg.get("seed", 42),
                         uncertain_text=dcfg.get("uncertain_text", "I am uncertain."))
    cut = int(0.85 * len(rows)); tr, va = rows[:cut], rows[cut:]
    log(f"[direct] rows={len(rows)} train={len(tr)} val={len(va)} counts={counts}")
    params = list(head.parameters()) + [p for p in backbone.parameters() if p.requires_grad] if head else \
        [p for p in backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.get("lr", 2e-5), weight_decay=cfg.get("weight_decay", 0.01))
    ga = cfg.get("grad_accum", 4); max_steps = cfg.get("max_steps")
    log_every = cfg.get("log_every", 1); max_len = cfg.get("max_len", 512); bs = cfg.get("batch_size", 4)
    ce = torch.nn.CrossEntropyLoss()
    step = 0; t0 = time.time()
    for ep in range(cfg.get("epochs", 3)):
        order = list(range(len(tr))); random.shuffle(order)
        for bi in range(0, len(order), bs):
            chunk = [tr[i] for i in order[bi:bi + bs]]
            loss_acc = 0; opt.zero_grad()
            for r in chunk:
                p = prompt_for(r["context"], r["question"], r["options"])
                if mode == "head":
                    enc = tok(p, return_tensors="pt", truncation=True, max_length=max_len)
                    enc = {k: v.to(device) for k, v in enc.items()}
                    h = backbone(**enc, output_hidden_states=True).hidden_states[-1][:, -1, :]
                    loss = ce(head(h.float()), torch.tensor([r["label"]], device=device))
                else:
                    full = p + " " + LETTERS[r["label"]]
                    enc = tok(full, return_tensors="pt", truncation=True, max_length=max_len + 4)
                    ids = enc["input_ids"].to(device); attn = enc["attention_mask"].to(device)
                    labels = ids.clone(); labels[:, :-1] = -100
                    loss = backbone(input_ids=ids, attention_mask=attn, labels=labels).loss
                (loss / len(chunk) / ga).backward()
                loss_acc += float(loss.detach())
            if (step + 1) % ga == 0: opt.step(); opt.zero_grad()
            if step % log_every == 0:
                log(f"[direct] ep={ep} step={step} loss={loss_acc/max(1,len(chunk)):.4f} elapsed={time.time()-t0:.1f}s")
            step += 1
            if max_steps and step >= max_steps: break
        if max_steps and step >= max_steps: break
        # per-epoch val on corrupted split (cheap, in-loop)
        backbone.eval(); correct = total = uh = ut = 0
        with torch.no_grad():
            for r in va:
                p = prompt_for(r["context"], r["question"], r["options"])
                enc = tok(p, return_tensors="pt", truncation=True, max_length=max_len)
                enc = {k: v.to(device) for k, v in enc.items()}
                if mode == "head":
                    h = backbone(**enc, output_hidden_states=True).hidden_states[-1][:, -1, :]
                    pr = int(head(h.float()).argmax(-1))
                else:
                    nxt = backbone(**enc).logits[0, -1]
                    vocab = tok.get_vocab()
                    ids = [(tok.convert_tokens_to_ids(" " + L) if (" " + L) in vocab else tok.convert_tokens_to_ids(L)) for L in LETTERS]
                    pr = int(torch.tensor(ids, device=device)[nxt[torch.tensor(ids, device=device)].argmax()])
                total += 1; correct += (pr == r["label"])
                if r["label"] == 4: ut += 1; uh += (pr == 4)
        log(f"[val ep={ep}] acc={correct/max(1,total):.3f} recall_unc={uh/max(1,ut):.3f} ({uh}/{ut})")
        backbone.train()
        step = 0
    backbone.eval(); correct = total = uh = ut = 0
    with torch.no_grad():
        for r in va:
            p = prompt_for(r["context"], r["question"], r["options"])
            enc = tok(p, return_tensors="pt", truncation=True, max_length=max_len)
            enc = {k: v.to(device) for k, v in enc.items()}
            if mode == "head":
                h = backbone(**enc, output_hidden_states=True).hidden_states[-1][:, -1, :]
                pr = int(head(h.float()).argmax(-1))
            else:
                nxt = backbone(**enc).logits[0, -1]
                vocab = tok.get_vocab()
                ids = [(tok.convert_tokens_to_ids(" " + L) if (" " + L) in vocab else tok.convert_tokens_to_ids(L)) for L in LETTERS]
                pr = int(torch.tensor(ids, device=device)[nxt[torch.tensor(ids, device=device)].argmax()])
            total += 1; correct += (pr == r["label"])
            if r["label"] == 4: ut += 1; uh += (pr == 4)
    log(f"[val] acc={correct/max(1,total):.3f} recall_unc={uh/max(1,ut):.3f} ({uh}/{ut})")
    # held-out clean eval, once at the end (SciQ validation, 4-way, no uncertainty)
    clean_acc, cc, tt = eval_clean_heldout(
        backbone, head, tok, dcfg, cfg, device, mode,
        seed=1234, cap=cfg.get("eval_clean_cap", 200))
    log(f"[eval-clean] acc={clean_acc:.3f} ({cc}/{tt})")
    backbone.save_pretrained(out / "lora")
    if head: torch.save(head.state_dict(), out / "head.pt")
    json.dump({"mode": mode, "acc": correct / max(1, total)}, open(out / "calib.json", "w"), indent=2)
    log("[direct] saved OK")

if __name__ == "__main__":
    main()
