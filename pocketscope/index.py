"""The pocket index and the MaxSim search over it."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

CHUNK = 20000


class PocketIndex:
    """The database of pocket tensors, searched by exact late-interaction MaxSim.

    Pass device="cuda" to hold the tensor on a GPU (22.9 GB for the released human index) or
    device="cpu" to score straight off the memory-mapped file, which is slower but needs no GPU.
    """

    def __init__(self, index_dir, device: str = "cuda", tensor="pocket_tensor.npy",
                 rows="pocket_index_rows.csv"):
        d = Path(index_dir)
        self.feats = np.load(d / tensor, mmap_mode="r")
        self.n, self.n_max, self.dim = self.feats.shape
        self.pocket_id, self.accession = [], []
        for r in csv.DictReader((d / rows).open()):
            self.pocket_id.append(r["pocket_id"])
            self.accession.append(r["uniprot_accession"])
        if len(self.pocket_id) != self.n:
            raise ValueError(f"{len(self.pocket_id)} rows described but tensor has {self.n}")
        self.device, self._T = device, None
        if device != "cpu":
            import torch
            self._torch = torch
            self._T = torch.empty((self.n, self.n_max, self.dim), dtype=torch.float16,
                                  device=device)
            for s in range(0, self.n, CHUNK):
                self._T[s:s + CHUNK] = torch.from_numpy(
                    np.array(self.feats[s:s + CHUNK])).to(device, torch.float16)

    def lengths(self) -> np.ndarray:
        """Residues per pocket, recovered from the zero padding."""
        out = np.zeros(self.n, np.int16)
        for s in range(0, self.n, CHUNK):
            blk = np.asarray(self.feats[s:s + CHUNK], np.float32)
            out[s:s + len(blk)] = (np.abs(blk).sum(2) > 0).sum(1)
        return out

    def score_all(self, query: np.ndarray) -> np.ndarray:
        """Asymmetric MaxSim of one (L, 1165) query against every pocket -> (n,) scores."""
        q = np.asarray(query, np.float32)
        if self.device == "cpu":
            out = np.empty(self.n, np.float32)
            for i in range(self.n):
                cand = np.asarray(self.feats[i], np.float32)
                keep = np.abs(cand).sum(1) > 0
                out[i] = (q @ cand[keep].T).max(1).mean() if keep.any() else -np.inf
            return out
        torch = self._torch
        Q = torch.from_numpy(q).to(self.device, torch.float16)
        out = torch.empty(self.n, device=self.device, dtype=torch.float32)
        for s in range(0, self.n, CHUNK):
            T = self._T[s:s + CHUNK]
            sims = torch.einsum("qd,nkd->nqk", Q, T).float()
            pad = (T.abs().sum(2) == 0).unsqueeze(1)
            out[s:s + T.shape[0]] = sims.masked_fill(pad, float("-inf")).max(2).values.mean(1)
        return out.cpu().numpy()

    def search(self, query: np.ndarray, top_k: int = 20, exclude=None):
        """-> [{rank, pocket_id, accession, score}], best pocket per protein, self excluded."""
        s = self.score_all(query)
        best = {}
        for i, acc in enumerate(self.accession):
            if exclude and acc == exclude:
                continue
            if acc not in best or s[i] > s[best[acc]]:
                best[acc] = i
        hits = sorted(best.values(), key=lambda i: -s[i])[:top_k]
        return [{"rank": r + 1, "pocket_id": self.pocket_id[i],
                 "accession": self.accession[i], "score": round(float(s[i]), 4)}
                for r, i in enumerate(hits)]
