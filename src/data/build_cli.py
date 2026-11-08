"""Thin CLI over jev_uncertain package (used by .bat runners)."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from jev_uncertain.dataset import load_sciq, build, save

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--no_model", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    ratios = {k: cfg.get(k, 0) for k in
              ["clean", "answer_absent_hard", "answer_absent_knn", "info_missing", "preserve"]}
    examples = load_sciq(n=(cfg.get("n_train", cfg.get("n_dry_run", 200)) if args.dry_run else None),
                         seed=cfg.get("seed", 42),
                         dataset_name=cfg.get("dataset_name", "allenai/sciq"))
    tok = model = None
    if not args.no_model:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        mn = cfg.get("model_name", "Qwen/Qwen3-0.6B")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[data] loading {mn} on {device}...")
        tok = AutoTokenizer.from_pretrained(mn, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            mn, trust_remote_code=True,
            torch_dtype=(torch.float16 if device.type == "cuda" else torch.float32))
        model.to(device)
        model.eval()
    rows, counts = build(examples, ratios, seed=cfg.get("seed", 42),
                         uncertain_text=cfg.get("uncertain_text", "I am uncertain."),
                         tok=tok, model=model,
                         max_hard=cfg.get("max_hard_score_samples", 200))
    fp = Path(args.out_dir) / ("jev_uncertain_dry.jsonl" if args.dry_run else "jev_uncertain_train.jsonl")
    save(rows, counts, fp)
    print("[data] wrote", fp, counts)

if __name__ == "__main__":
    main()
