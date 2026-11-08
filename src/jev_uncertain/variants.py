"""Variant builders: one function per corruption. Each takes rng + inputs, returns
(opts4, gold_index_or_None, label, detail). U is appended by dataset builder, fixed last."""
import random
from .corrupt import sent_split, drop_key_sentence, rule_paraphrase, knn_distractor, score_top_wrong

def v_clean(e, rng):
    opts4 = rng.sample([e["correct_answer"]] + e["distractors"], 4)
    gi = opts4.index(e["correct_answer"])
    return opts4, e["support"] or "", e["question"], gi, gi, {}

def v_answer_absent_hard(e, i, examples, M, vec, rng, tok=None, model=None, max_hard=200):
    gold, distract = e["correct_answer"], list(e["distractors"])
    if tok is not None and model is not None and i < max_hard:
        hard, det = score_top_wrong(e["question"], distract, tok, model)
    else:
        hard, det = rng.choice(distract), {"fallback": True}
    extra = None
    try:
        if M is not None:
            extra, _ = knn_distractor(i, examples, M, vec)
    except Exception:
        extra = None
    cand = [hard] + distract + ([extra] if extra else [])
    pool = [o for o in dict.fromkeys(cand) if o.strip().lower() != gold.strip().lower()]
    while len(pool) < 4:
        pool.append(rng.choice(distract))
    return rng.sample(pool[:4], 4), e["support"] or "", e["question"], None, 4, {"hard": hard, **det}

def v_answer_absent_knn(e, i, examples, M, vec, rng):
    gold, distract = e["correct_answer"], list(e["distractors"])
    if M is not None:
        borrowed, det = knn_distractor(i, examples, M, vec)
    else:
        borrowed, det = rng.choice(distract), {"fallback": True}
    pool = [o for o in dict.fromkeys([borrowed] + distract) if o.strip().lower() != gold.strip().lower()]
    while len(pool) < 4:
        pool.append(rng.choice(distract))
    return rng.sample(pool[:4], 4), e["support"] or "", e["question"], None, 4, det

def v_info_missing(e, rng):
    src = e["support"] or ""
    if not src.strip():
        # source already empty: nothing to remove, mark it
        opts4 = rng.sample([e["correct_answer"]] + e["distractors"], 4)
        return opts4, "", e["question"], None, 4, {"strategy": "already_empty_in_source"}
    ctx, det = drop_key_sentence(src, e["question"], e["correct_answer"])
    opts4 = rng.sample([e["correct_answer"]] + e["distractors"], 4)
    return opts4, ctx, e["question"], None, 4, det or {}

def v_preserve(e, examples, rng):
    q = rule_paraphrase(e["question"])
    other = rng.choice(examples)["support"] or ""
    pre = (sent_split(other)[:1] + [""])[0]
    ctx = ((pre + " " + (e["support"] or "")).strip() if pre else (e["support"] or ""))
    opts4 = rng.sample([e["correct_answer"]] + e["distractors"], 4)
    gi = opts4.index(e["correct_answer"])
    return opts4, ctx, q, gi, gi, {"noisy": bool(pre)}
