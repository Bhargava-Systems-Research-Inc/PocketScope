"""Frozen ESM-C 600M, used exactly as released. Nothing here is trained."""
from __future__ import annotations
import numpy as np

MODEL = "esmc-600m-2024-12"
_state = {"model": None, "tok": None}


def _load(device: str):
    if _state["model"] is None:
        import torch
        from esm.pretrained import ESMC_600M_202412
        from esm.tokenization import get_esmc_model_tokenizers
        _state["model"] = ESMC_600M_202412(device=torch.device(device)).eval()
        _state["tok"] = get_esmc_model_tokenizers()
    return _state["tok"], _state["model"]


def embed_chain(seq: str, device: str = "cuda") -> np.ndarray:
    """(len(seq), 1152) last-hidden-layer per-residue embeddings for one chain."""
    import torch
    tok, model = _load(device)
    enc = tok([seq], return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model.forward(sequence_tokens=enc["input_ids"].to(device),
                            sequence_id=enc["attention_mask"].to(device))
    return out.embeddings[0, 1:1 + len(seq)].float().cpu().numpy()   # strip BOS/EOS
