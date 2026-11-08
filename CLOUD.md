# Cloud (GPU) quickstart — run on a fresh VM (e.g. Lambda/RunPod/Vast, 1x A10 or T4)
# Code is identical; only torch source + scale knobs change.

# 1. Get code (pick one; do NOT zip):
#   git init && git add -A && git commit -m init && git push <your-repo>
#   # or: gh repo create llm_jev --private --source=. --push
# On VM:
#   git clone <your-repo-url> && cd LLM_JEV

# 2. Env (conda or venv):
conda create -n llm_jev python=3.10 -y && conda activate llm_jev
# GPU torch (CUDA 12.1):
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-gpu.txt

# 3. Optional login (raises HF rate limits, needed for gated models — Qwen3-0.6B is open):
# huggingface-cli login

# 4. Dataset (full, with base-model hard negatives on GPU = fast):
python src/data/build_cli.py --config configs/data.yaml --out_dir data
# ^ no --dry_run flag => full ~10.5k rows. Ratios stay as-is in data.yaml.
# Dry-run locally instead: add --dry_run (uses n_train: 200 if set, else n_dry_run).

# 5. Train (GPU): override the CPU dry-run caps.
# Option A: edit configs/train.yaml -> batch_size: 8, grad_accum: 2, max_steps: null, epochs: 3
# Option B: one-off env overrides not supported; just edit the yaml.
python src/jev_uncertain/train.py --config configs/train.yaml --data_config configs/data.yaml

# 6. Artifacts: outputs/jev_run/{train.log,head.pt,calib.json} + data/*.jsonl
# Copy back: scp <vm>:LLM_JEV/outputs/jev_run/* ./outputs/jev_run/

# Suggested GPU config (A10 24GB, Qwen3-0.6B full-train starting point):
# batch_size: 8, grad_accum: 2, epochs: 3, lr: 2.0e-4, max_steps: null, max_len: 512
# Scale-up later: Qwen3-4B/8B needs batch_size 2-4 + grad_accum 8-16 on 24GB.
