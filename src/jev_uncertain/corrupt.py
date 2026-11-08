"""Corruption primitives: render, sent_split, drop_key_sentence, rule_paraphrase,
build_tfidf_index, knn_distractor, score_top_wrong. Pure functions, importable."""
import re

PROMPT_TMPL = (
    "Context: {context}\nQuestion: {question}\nOptions:\n"
    "A) {o0}\nB) {o1}\nC) {o2}\nD) {o3}\nE) {o4}\nAnswer with one of A,B,C,D,E."
)
LABELS = "ABCDE"

def render(context, question, options):
    prompt = PROMPT_TMPL.format(context=context, question=question,
        o0=options[0], o1=options[1], o2=options[2], o3=options[3], o4=options[4])
    spans = []
    for i, opt in enumerate(options):
        letter = LABELS[i]
        needle = f"{letter}) {opt}"
        s = prompt.find(needle) + len(f"{letter}) ")
        spans.append([s, s + len(opt)])
    return prompt, spans

def sent_split(t):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', (t or "").strip()) if s.strip()]

def drop_key_sentence(support, question, gold):
    sents = sent_split(support or "")
    if not sents:
        return "", None
    qg = set((question + " " + gold).lower().split())
    def overlap(s):
        return len(set(s.lower().split()) & qg)
    scores = [overlap(s) for s in sents]
    if max(scores) == 0:
        return "", {"strategy": "drop_all_no_overlap"}
    idx = int(max(range(len(sents)), key=lambda i: scores[i]))
    kept = [s for j, s in enumerate(sents) if j != idx]
    return " ".join(kept), {"strategy": "drop_max_overlap", "dropped": sents[idx]}

def rule_paraphrase(t):
    return re.sub(r'\s+', ' ', (t or "")).strip()

def build_tfidf_index(examples):
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = [e["question"] + " " + e["support"] for e in examples]
    vec = TfidfVectorizer(max_features=20000, stop_words="english").fit(docs)
    M = vec.transform(docs)
    return vec, M

def knn_distractor(i, examples, M, vec):
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(M[i], M).ravel()
    sims[i] = -1
    order = sims.argsort()[::-1]
    my_gold = examples[i]["correct_answer"].strip().lower()
    for j in order[:50]:
        cand = examples[j]["correct_answer"]
        if cand.strip().lower() != my_gold:
            return cand, {"nn_idx": int(j), "sim": float(sims[j])}
    return examples[(i + 1) % len(examples)]["correct_answer"], {"nn_idx": -1, "sim": 0.0}

def score_top_wrong(question, distractors, tok, model):
    import torch
    dev = next(model.parameters()).device
    best, best_score = distractors[0], None
    with torch.no_grad():
        for d in distractors:
            txt = f"Question: {question} Answer: {d}."
            ids = tok(txt, return_tensors="pt")["input_ids"].to(dev)
            out = model(ids, labels=ids)
            nll = float(out.loss)
            if best_score is None or nll < best_score:
                best_score, best = nll, d
    return best, {"nll": best_score}
