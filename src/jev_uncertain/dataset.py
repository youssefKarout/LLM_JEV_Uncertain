"""Dataset assembly: load sciq, sample variants per ratios, render + spans, write JSONL."""
import json, random
from pathlib import Path
from .corrupt import render, build_tfidf_index, LABELS
from . import variants as V

FN_MAP = {
    "clean": "clean", "answer_absent_hard": "hard",
    "answer_absent_knn": "knn", "info_missing": "missing", "preserve": "preserve",
}

def load_sciq(n=None, seed=42, dataset_name="allenai/sciq", split="train",
             min_support_chars=20):
    from datasets import load_dataset
    rng = random.Random(seed)
    ds = load_dataset(dataset_name, split=split)
    rows = [{"question": r["question"], "correct_answer": r["correct_answer"],
             "distractors": [r["distractor1"], r["distractor2"], r["distractor3"]],
             "support": r.get("support", "") or ""} for r in ds]
    n_empty = sum(1 for r in rows if len(r["support"].strip()) < min_support_chars)
    print(f"[load] {len(rows)} rows, {n_empty} with support<{min_support_chars} chars "
          f"({n_empty/max(1,len(rows)):.1%}) — these are EMPTY IN SOURCE (sciq), not our bug")
    rows_kept = [r for r in rows if len(r["support"].strip()) >= min_support_chars]
    print(f"[load] kept {len(rows_kept)} rows with non-empty support for generation")
    if n is not None:
        rows_kept = rng.sample(rows_kept, min(n, len(rows_kept)))
    return rows_kept

def build(examples, ratios, seed=42, uncertain_text="I am uncertain.", tok=None, model=None,
          max_hard=200, tfidf=True):
    rng = random.Random(seed)
    names = ["clean", "answer_absent_hard", "answer_absent_knn", "info_missing", "preserve"]
    weights = [ratios.get(k, 0) for k in names]
    vec = M = None
    if tfidf:
        try:
            vec, M = build_tfidf_index(examples)
        except Exception:
            vec, M = None, None
    rows, counts = [], {k: 0 for k in names}
    for i, e in enumerate(examples):
        v = rng.choices(names, weights=weights)[0]
        counts[v] += 1
        if v == "clean":
            opts4, ctx, q, gi, label, det = V.v_clean(e, rng)
        elif v == "answer_absent_hard":
            opts4, ctx, q, gi, label, det = V.v_answer_absent_hard(
                e, i, examples, M, vec, rng, tok, model, max_hard)
        elif v == "answer_absent_knn":
            opts4, ctx, q, gi, label, det = V.v_answer_absent_knn(e, i, examples, M, vec, rng)
        elif v == "info_missing":
            opts4, ctx, q, gi, label, det = V.v_info_missing(e, rng)
        else:
            opts4, ctx, q, gi, label, det = V.v_preserve(e, examples, rng)
        options = opts4 + [uncertain_text]
        prompt, spans = render(ctx, q, options)
        rows.append({"id": f"sciq-{i}-{v}", "variant": v, "prompt": prompt,
            "context": ctx, "question": q, "options": options,
            "gold_index": gi, "label": int(label), "label_name": LABELS[label],
            "option_spans": spans, "position_ids": None,
            "meta": {"corruption_detail": det}})
    return rows, counts

def save(rows, counts, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    priors = {"counts": counts,
              "p_uncertain": (counts["answer_absent_hard"] + counts["answer_absent_knn"]
                              + counts["info_missing"]) / max(1, len(rows))}
    with open(out_path.parent / "label_priors.json", "w") as f:
        json.dump(priors, f, indent=2)
    return priors
