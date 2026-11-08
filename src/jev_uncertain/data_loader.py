"""Dataset + collator: tokenize special-token layout, pool ids, option mask,
position_ids=None default (future RoPE patch plugs in here)."""
import torch
from torch.utils.data import Dataset
from .tokens import add_special_tokens, render_ids, option_mask, UNC_E, OPT_E, ANS

class JEVDataset(Dataset):
    def __init__(self, rows, tok, max_len=512, unc_text=None):
        self.rows, self.tok, self.max_len = rows, tok, max_len
        self.unc = unc_text
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        opts4 = r["options"][:4]
        unc = r["options"][4] if self.unc is None else self.unc
        text = render_ids(r["context"], r["question"], opts4, unc)
        enc = self.tok(text, truncation=True, max_length=self.max_len)
        return {"input_ids": enc["input_ids"],
                "label": int(r["label"]),
                "n_opts": min(4, len(opts4)),
                "row_id": r["id"]}

def _find(tok, ids, piece):
    pid = tok.convert_tokens_to_ids(piece)
    return [j for j, x in enumerate(ids) if x == pid]

def make_collator(tok, max_len=512, position_mode="default"):
    """position_mode: 'default' => position_ids=None (model default causal-pos).
    'flat_options' (future) => same position id across option spans (needs span map)."""
    pad = tok.pad_token_id
    def collate(batch):
        ids = [torch.tensor(b["input_ids"][:max_len]) for b in batch]
        L = max(len(x) for x in ids)
        inp = torch.stack([torch.cat([x, torch.full((L - len(x),), pad)]) for x in ids])
        attn = (inp != pad).long()
        labels = torch.tensor([b["label"] for b in batch])
        # pool positions: last [/OPT] per option slot, [/UNC], [ANS]
        opt_e, unc_e, ans_id = tok.convert_tokens_to_ids(OPT_E), tok.convert_tokens_to_ids(UNC_E), tok.convert_tokens_to_ids(ANS)
        B = len(batch)
        pool = torch.zeros(B, 6, dtype=torch.long)  # 4 opt + unc + ans
        omask = torch.zeros(B, 5)
        for b in range(B):
            seq = inp[b].tolist()
            opos = [j for j, x in enumerate(seq) if x == opt_e][:4]
            upos = [j for j, x in enumerate(seq) if x == unc_e]
            apos = [j for j, x in enumerate(seq) if x == ans_id]
            for k in range(4):
                pool[b, k] = opos[k] if k < len(opos) else 0
            pool[b, 4] = upos[-1] if upos else 0
            pool[b, 5] = apos[-1] if apos else (attn[b].sum().item() - 1)
            omask[b] = torch.tensor(option_mask(batch[b]["n_opts"]))
        out = {"input_ids": inp, "attention_mask": attn, "labels": labels,
               "pool_idx": pool, "option_mask": omask}
        out["position_ids"] = None  # hook: Block 2 RoPE patch fills this
        return out
    return collate
