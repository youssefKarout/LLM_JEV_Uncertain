"""Model: Qwen backbone + set-attention over [4 opt + unc] pooled states + 5-way head.
Forward returns logits [B,5]. Bidirectional note: Qwen is causal; true full
bidirectional needs attn-mask surgery (Block 2 cloud). For now: causal backbone
+ set-attention across options compensates order effects; augmentation covers the rest.
"""
import torch
import torch.nn as nn

class SetHead(nn.Module):
    def __init__(self, hidden, n_opts=5, heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(hidden)
        self.cls = nn.Linear(hidden, 1)
    def forward(self, feats, option_mask):
        # feats may be bf16 from backbone — cast to head dtype
        feats = feats.to(self.cls.weight.dtype)
        # feats [B,5,H], option_mask [B,5] 1=real 0=pad
        key_pad = (option_mask == 0)  # True = ignore
        a, _ = self.attn(feats, feats, feats, key_padding_mask=key_pad)
        h = self.ln(feats + a)
        return self.cls(h).squeeze(-1)  # [B,5]

class JEVClassifier(nn.Module):
    def __init__(self, backbone, hidden, heads=4, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.set = SetHead(hidden, heads=heads, dropout=dropout)
    def forward(self, input_ids, attention_mask=None, pool_idx=None,
                option_mask=None, position_ids=None, labels=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, output_hidden_states=True)
        # CausalLM outputs logits; pool from last hidden layer
        if hasattr(out, "last_hidden_state"):
            last = out.last_hidden_state
        else:
            last = out.hidden_states[-1]  # [B,L,H]
        B = last.size(0)
        # pool: [B,6] -> first 5 are opt0-3 + unc; ans at [:,5] reserved as query later
        gathered = last[torch.arange(B).unsqueeze(1), pool_idx[:, :5].clamp(0, last.size(1)-1)]
        logits = self.set(gathered, option_mask)
        # mask padded options to -inf so they can't win
        logits = logits + (1 - option_mask) * -1e9
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return {"logits": logits, "loss": loss}
