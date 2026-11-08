"""Special tokens for JEV+Uncertain single-seq classifier.
Layout: [CTX] ctx [/CTX] [Q] q [/Q] [OPT] o0 [/OPT] ... [UNC] uncertain [/UNC] [ANS]
Pooling: hidden at each [/OPT_i], [/UNC], (+ [ANS] as query).
Variable options: pad missing content options with [PADOPT], mask them in set-attention + loss.
"""
SPECIAL_TOKENS = [
    "[CTX]", "[/CTX]",
    "[Q]", "[/Q]",
    "[OPT]", "[/OPT]",
    "[UNC]", "[/UNC]",
    "[ANS]",
    "[PADOPT]",
]
CTX_S, CTX_E, Q_S, Q_E, OPT_S, OPT_E, UNC_S, UNC_E, ANS, PADOPT = SPECIAL_TOKENS

N_CONTENT = 4  # A-D slots; slot i masked if example has fewer
N_TOTAL = 5    # 4 content + uncertain

def add_special_tokens(tok):
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    return tok

def render_ids(ctx, q, options4, unc_text):
    """Return text with special tokens. options4: list len<=4 (pad later)."""
    parts = [f"{CTX_S} {ctx} {CTX_E} ", f"{Q_S} {q} {Q_E} "]
    for o in options4:
        parts.append(f"{OPT_S} {o} {OPT_E} ")
    for _ in range(N_CONTENT - len(options4)):
        parts.append(f"{OPT_S} {PADOPT} {OPT_E} ")
    parts.append(f"{UNC_S} {unc_text} {UNC_E} {ANS}")
    return "".join(parts)

def option_mask(n_real_options):
    """[5] float mask: 1 for real content options + uncertain, 0 for padded."""
    m = [1.0 if i < n_real_options else 0.0 for i in range(N_CONTENT)] + [1.0]
    return m
