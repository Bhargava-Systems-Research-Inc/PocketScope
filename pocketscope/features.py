"""The pocket representation: Equation 1 of the paper.

A pocket of L residues becomes an (L, 1165) float array. Each residue contributes 1152
ESM-C 600M dimensions and 13 physicochemical channels. The two blocks are L2-normalised
separately, the chemical block is scaled by ALPHA, and the concatenation is renormalised, so
every residue vector has unit norm and a dot product between two of them is a cosine.
"""
from __future__ import annotations
import numpy as np

ESM_DIM, PHYS_DIM, FEAT_DIM = 1152, 13, 1165
ALPHA = 0.5          # weight of the chemical block, fixed before any benchmarking
N_MAX = 64           # residues kept per pocket, in P2Rank rank order

# Kyte-Doolittle hydropathy, formal charge, side-chain volume, aromaticity,
# side-chain H-bond donors, acceptors, polarity.
_PHYS = {
    "A": (1.8, 0, 88.6, 0, 0, 0, 0),   "R": (-4.5, 1, 173.4, 0, 5, 0, 1),
    "N": (-3.5, 0, 114.1, 0, 2, 2, 1), "D": (-3.5, -1, 111.1, 0, 0, 4, 1),
    "C": (2.5, 0, 108.5, 0, 1, 0, 0),  "Q": (-3.5, 0, 143.8, 0, 2, 2, 1),
    "E": (-3.5, -1, 138.4, 0, 0, 4, 1),"G": (-0.4, 0, 60.1, 0, 0, 0, 0),
    "H": (-3.2, 0, 153.2, 1, 1, 1, 1), "I": (4.5, 0, 166.7, 0, 0, 0, 0),
    "L": (3.8, 0, 166.7, 0, 0, 0, 0),  "K": (-3.9, 1, 168.6, 0, 3, 0, 1),
    "M": (1.9, 0, 162.9, 0, 0, 1, 0),  "F": (2.8, 0, 189.9, 1, 0, 0, 0),
    "P": (-1.6, 0, 112.7, 0, 0, 0, 0), "S": (-0.8, 0, 89.0, 0, 1, 1, 1),
    "T": (-0.7, 0, 116.1, 0, 1, 1, 1), "W": (-0.9, 0, 227.8, 1, 1, 0, 0),
    "Y": (-1.3, 0, 193.6, 1, 1, 1, 1), "V": (4.2, 0, 140.0, 0, 0, 0, 0),
}
_NORM = np.array([4.5, 1.0, 230.0, 1.0, 5.0, 4.0, 1.0], dtype=np.float32)

# 6 multi-hot pharmacophore bits: positive, negative, polar, aromatic, hydrophobic, other.
TYPE_NAMES = ["POS", "NEG", "POLAR", "AROMATIC", "HYDROPHOBIC", "OTHER"]
_AA_TYPES = {
    "R": (0,), "K": (0,), "H": (0, 3), "D": (1,), "E": (1,),
    "S": (2,), "T": (2,), "N": (2,), "Q": (2,), "Y": (2, 3),
    "F": (3,), "W": (3,),
    "A": (4,), "V": (4,), "L": (4,), "I": (4,), "M": (4,), "C": (4,), "P": (4,), "G": (4,),
}


def residue_phys(seq: str) -> np.ndarray:
    """(L, 13): seven normalised scalars followed by six pharmacophore bits."""
    scal = np.zeros((len(seq), 7), np.float32)
    types = np.zeros((len(seq), 6), np.float32)
    for i, aa in enumerate(seq):
        if aa in _PHYS:
            scal[i] = _PHYS[aa]
        for k in _AA_TYPES.get(aa, (5,)):
            types[i, k] = 1.0
    return np.concatenate([scal / _NORM, types], axis=1)


def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def match_feature(esm: np.ndarray, phys: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Equation 1: normalize([normalize(esm) || alpha * normalize(phys)]) -> (L, 1165)."""
    r = np.concatenate([_unit(np.asarray(esm, np.float32)),
                        alpha * _unit(np.asarray(phys, np.float32))], axis=-1)
    return _unit(r).astype(np.float32)


def encode_pocket(pocket_seq: str, esm_rows: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Pocket one-letter sequence + its ESM rows -> the (L, 1165) pocket tensor."""
    if len(pocket_seq) != len(esm_rows):
        raise ValueError(f"{len(pocket_seq)} residues but {len(esm_rows)} ESM rows")
    return match_feature(esm_rows, residue_phys(pocket_seq), alpha)
