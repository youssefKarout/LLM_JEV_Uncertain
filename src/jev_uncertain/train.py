"""Training loop: LoRA on Qwen backbone (frozen) + trainable SetHead.
Plain cross-entropy. CPU-safe dry-run. Saves head + LoRA + temp scaling."""
import argparse, json, sys, random
from pathlib import Path
import torch
import yaml
sys.path.insert(0, "src")
from jev_uncertain.tokens import add_special_tokens
from jev_uncertain.dataset import load_sciq, build
from jev_uncertain.data_loader import JEVDataset, make_collator
from jev_uncertain.model import JEVClassifier

def fit_temperature(logits, labels):
    import torch.nn as nn
    T = torch.ones(1, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50)
    ce = nn.CrossEntropyLoss()
    def closure():
        opt.zero_grad()
        loss = ce(logits / T.clamp(0.05, 10.0), labels)
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.detach().clamp(0.05, 10.0))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--data_config", default="configs/data.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    dcfg = yaml.safe_load(open(args.data_config))
    torch.manual_seed(cfg.get("seed", 42)); random.seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda"
    out = Path(cfg.get("out_dir", "outputs/jev_run")); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "w")
    def log(*a):
        print(*a); print(*a, file=logf, flush=True)
    log(f"[train] device={device} fp16={use_fp16}")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    mn = cfg.get("model_name", "Qwen/Qwen3-0.6B")
    log(f"[train] load {mn} on {device}")
    tok = AutoTokenizer.from_pretrained(mn, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    add_special_tokens(tok)
    backbone = AutoModelForCausalLM.from_pretrained(
        mn, trust_remote_code=True,
        torch_dtype=(torch.float16 if use_fp16 else torch.float32))
    backbone.resize_token_embeddings(len(tok))
    try:
        from peft import LoraConfig, get_peft_model
        lc = LoraConfig(r=cfg.get("lora_r", 16), lora_alpha=cfg.get("lora_alpha", 32),
                        lora_dropout=cfg.get("lora_dropout", 0.05),
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                        bias="none", task_type="CAUSAL_LM")
        backbone = get_peft_model(backbone, lc)
        backbone.print_trainable_parameters()
    except ImportError as e:
        log(f"[train] peft missing ({e}) — install peft for LoRA; freezing backbone instead")
        for p in backbone.parameters(): p.requires_grad = False
    clf = JEVClassifier(backbone, backbone.config.hidden_size,
                        heads=cfg.get("heads", 4), dropout=cfg.get("head_dropout", 0.1))
    clf.to(device)
    clf.train()

    ratios = {k: dcfg.get(k, 0) for k in
              ["clean", "answer_absent_hard", "answer_absent_knn", "info_missing", "preserve"]}
    n_train = dcfg.get("n_train", dcfg.get("n_dry_run", 200))  # null = full
    ex = load_sciq(n=n_train, seed=cfg.get("seed", 42),
                   dataset_name=dcfg.get("dataset_name", "allenai/sciq"))
    rows, counts = build(ex, ratios, seed=cfg.get("seed", 42),
                         uncertain_text=dcfg.get("uncertain_text", "I am uncertain."))
    cut = int(0.85 * len(rows))
    tr, va = rows[:cut], rows[cut:]
    log(f"[train] rows={len(rows)} train={len(tr)} val={len(va)} counts={counts}")
    col = make_collator(tok, max_len=cfg.get("max_len", 512))
    tr_ds, va_ds = JEVDataset(tr, tok, cfg.get("max_len", 512)), JEVDataset(va, tok, cfg.get("max_len", 512))
    from torch.utils.data import DataLoader
    bs = cfg.get("batch_size", 2)
    tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True, collate_fn=col)
    va_dl = DataLoader(va_ds, batch_size=bs, collate_fn=col)

    opt = torch.optim.AdamW([p for p in clf.parameters() if p.requires_grad],
                            lr=cfg.get("lr", 2e-4), weight_decay=cfg.get("weight_decay", 0.01))
    ga = cfg.get("grad_accum", 8)
    max_steps = cfg.get("max_steps", 20)
    log_every = cfg.get("log_every", 1)  # log every N minibatches
    log(f"[train] max_steps={max_steps} log_every={log_every} grad_accum={ga}")
    import time
    import torch.nn.functional as _F
    t0 = time.time()
    step = 0
    def run_val(tag):
        clf.eval(); all_l, all_y = [], []
        with torch.no_grad():
            for b in va_dl:
                bb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
                o = clf(**{k: bb[k] for k in ("input_ids", "attention_mask", "pool_idx", "option_mask", "position_ids")})
                all_l.append(o["logits"].cpu()); all_y.append(bb["labels"].cpu())
        L = torch.cat(all_l); Y = torch.cat(all_y)
        T = fit_temperature(L, Y)
        pred = (L / T).argmax(-1)
        acc = float((pred == Y).float().mean())
        p_unc = _F.softmax(L / T, -1)[:, 4]
        thr = cfg.get("uncal_threshold", 0.5)
        flag = p_unc > thr
        is_unc = Y == 4
        log(f"[val {tag}] T={T:.3f} acc={acc:.3f} flag_rate={float(flag.float().mean()):.3f} "
            f"recall_unc={float((flag & is_unc).sum() / max(1, int(is_unc.sum()))):.3f}")
        clf.train()
        return T, acc, thr
    for ep in range(cfg.get("epochs", 1)):
        for b in tr_dl:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            out_b = clf(**{k: b[k] for k in ("input_ids", "attention_mask", "pool_idx", "option_mask", "position_ids")}, labels=b["labels"])
            (out_b["loss"] / ga).backward()
            if (step + 1) % ga == 0:
                opt.step(); opt.zero_grad()
            if step % log_every == 0:
                dt = time.time() - t0
                log(f"[train] ep={ep} step={step} loss={out_b['loss'].detach().float().item():.4f} elapsed={dt:.1f}s")
            step += 1
            if max_steps and step >= max_steps:
                break
        T, acc, thr = run_val(f"ep={ep}")
        if max_steps and step >= max_steps:
            break
        step = 0  # per-epoch step reset (your fix, kept)
    opt.zero_grad()
    # val: logits for temp scaling + uncertain metrics
    T, acc, thr = run_val("final")
    # held-out clean eval, once at the end (SciQ validation, 4-way, no uncertainty)
    try:
        from jev_uncertain.train_direct import eval_clean_heldout, LETTERS  # noqa
        from jev_uncertain.train_direct import prompt_for as _pf  # noqa (ensures import ok)
        clean_acc, cc, tt = eval_clean_heldout(
            backbone, None, tok, dcfg, cfg, device, "gen",
            seed=1234, cap=cfg.get("eval_clean_cap", 200))
        # NOTE: SetHead model uses special-token layout; letter-logit eval above is a
        # backbone-only proxy. Full head eval on clean rows = cloud follow-up.
        log(f"[eval-clean] acc={clean_acc:.3f} ({cc}/{tt}) [backbone letter-logit proxy]")
    except Exception as e:
        log(f"[eval-clean] skipped ({e})")
    torch.save({"set": clf.set.state_dict(), "T": T,
                "lora": backbone.state_dict() if hasattr(backbone, "peft_config") else None},
               out / "head.pt")
    json.dump({"T": T, "acc": acc, "threshold": thr}, open(out / "calib.json", "w"), indent=2)
    log("[train] saved head.pt + calib.json OK")

if __name__ == "__main__":
    main()
