# LLM_JEV — JEV + Uncertain (Block 1 dataset + Block 2 training)

JEV-style scorer: input = prompt + context + question + options, output = logits[5] → softmax → scores 0–100 over `[A,B,C,D,UNCERTAIN]`. Trained with plain cross-entropy (LoRA + trainable head). Label 4 = UNCERTAIN.

Two training scripts share one dataset builder:

- `src/jev_uncertain/train.py` — SetHead architecture (special-token layout `[CTX]/[/CTX] [Q]/[/Q] [OPT]/[/OPT]…`, set-attention over pooled option states).
- `src/jev_uncertain/train_direct.py` — direct Qwen finetune: `mode: head` (last-token pooling + Linear 5-way) or `mode: gen` (LM finetune emitting the answer letter, CE only on the final token).

## Setup (conda, CPU)

```bat
conda env create -f env.yml
conda activate llm_jev_cpu
python -m pip install -r requirements-cpu.txt   # torch CPU + transformers + datasets + sklearn + peft (+ torchao>=0.16 for peft)
```

GPU cloud: `pip install torch --index-url https://download.pytorch.org/whl/cu121 && pip install -r requirements-gpu.txt` (see `CLOUD.md` for clone→train→scp-back flow).

## Model download (Qwen3-0.6B)

```bat
huggingface-cli download Qwen/Qwen3-0.6B
```

## Block 1 — dataset corruption (SciQ → JEV+Uncertain)

Source: `allenai/sciq` (`question`, `correct_answer`, `distractor1-3`, `support` = context). Rows with empty/short `support` (<20 chars) are dropped — they're empty upstream, not a corruption bug.

Variants, sampled per `configs/data.yaml` ratios (default clean .65 / hard .08 / knn .07 / missing .10 / preserve .10):

| Variant | Corruption | Target label | How |
|---|---|---|---|
| `clean` | none | gold index 0-3 | gold + 3 distractors, shuffled A-D |
| `answer_absent_hard` | replace gold with base-Qwen top-wrong distractor (lowest NLL of "Q? A." on CPU, capped sample) | 4 = UNCERTAIN | gold absent; 4th option borrowed via KNN if needed |
| `answer_absent_knn` | replace gold with correct answer from TF-IDF nearest-neighbor SciQ row | 4 = UNCERTAIN | gold absent; hard distractor from similar prompt |
| `info_missing` | drop max-overlap sentence from `context` (or all context if no overlap) | 4 = UNCERTAIN | gold option present but its supporting fact removed |
| `preserve` | rule paraphrase + irrelevant context prefix | gold index 0-3 | answer-preserving perturbation |

- Options A-D are **uniformly shuffled**; `gold_index` remapped. UNCERTAIN (`I am uncertain — none of the above options is satisfactory.`) is **fixed last** (E), never shuffled — no per-option prompts.
- `option_spans` (char offsets of each option in the rendered prompt) saved for future custom `position_ids` work; `position_ids=null` by default (model default causal positions).
- `label_priors.json` written with `p_uncertain` for temperature scaling / prior shift on held-out at real base rate.

Generate (dry run, no model scoring, fastest):

```bat
python src/data/build_cli.py --config configs/data.yaml --out_dir data --dry_run --no_model
```

With base-model top-wrong hard negatives (CPU, slower; set `USE_BASE_MODEL` in the notebook):

```bat
python src/data/build_cli.py --config configs/data.yaml --out_dir data --dry_run
```

Full dataset (omit `--dry_run`; uses `n_train: null` = all support-filtered rows):

```bat
python src/data/build_cli.py --config configs/data.yaml --out_dir data
```

The notebook `notebooks/generate_variants.ipynb` is the generator frontend — it calls the package fns directly (`corrupt.py`, `variants.py`, `dataset.py`) so you can inspect single-example demos before the full build.

## Outputs

- `data/jev_uncertain_dry.jsonl` / `jev_uncertain_train.jsonl`
- `data/label_priors.json` (p_uncertain for later calibration)
- Each row: prompt, context, question, options[5] (U fixed last), gold_index (null if uncertain), label 0-4, option_spans (char offsets for future custom position_ids), position_ids=null (collator fills later).

## Block 2 — training

`configs/train.yaml` (SetHead): `batch_size`, `grad_accum`, `epochs`, `lr`, `lora_r/alpha/dropout` on `q/k/v/o_proj`, `max_steps` (null = full), `uncal_threshold` for uncertain flagging, `eval_clean_cap` for held-out clean eval rows per end-of-run.

`configs/train_direct.yaml` (direct): same knobs + `mode: head | gen`, `eval_clean_cap`.

Run:

```bat
python src/jev_uncertain\train.py --config configs\train.yaml --data_config configs\data.yaml
python src/jev_uncertain\train_direct.py --config configs\train_direct.yaml --data_config configs\data.yaml
```

Training loop:

1. Qwen3-0.6B (float16 on CUDA, float32 on CPU) + LoRA + special-token resize.
2. Build dataset with chosen ratios; 85/15 train/val split.
3. **Per epoch**: train minibatches (grad accumulation) → `[val ep=N]` on corrupted val split (acc / recall_unc / flag_rate / temperature T refit).
4. **End of run**: `[eval-clean]` once on held-out SciQ validation (clean 4-way, uncertain slot ignored).
5. Save `head.pt` (SetHead state + T) + `lora/` adapter + `calib.json`.

Loss: plain cross-entropy only — no exotic losses.

## Eval a trained checkpoint

```bat
python src/jev_uncertain\eval_head.py --ckpt loacl\head.pt
```

Reports CLEAN 4-way acc (gold among A-D, uncertain ignored) + CORRUPT 5-way acc / recall_unc.

## Evaluation notes

- Output is **logits/softmax over 5 choices**, not text generation (unless `mode: gen`).
- No causal masking inside options — set-attention over pooled option states is order-free; shuffle augmentation covers position invariance.
- `uncertain` triggering at inference: `argmax == 4` or `p(UNCERTAIN) > uncal_threshold` (tuned with temperature scaling / prior shift on held-out at your real base rate).
