"""Reading a pocket out of a PDB file and a P2Rank prediction."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

AA3 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
       "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
       "THR":"T","TRP":"W","TYR":"Y","VAL":"V","MSE":"M","SEC":"U","UNK":"X"}


def parse_ca(pdb: str | Path) -> dict:
    """-> {(chain, resnum): (one-letter, xyz)} in file order, CA atoms of standard residues."""
    out = {}
    for line in Path(pdb).read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")) or line[12:16].strip() != "CA":
            continue
        rn = line[17:20].strip()
        if rn not in AA3:
            continue
        key = (line[21].strip() or "_", int(line[22:26]))
        if key in out:
            continue
        out[key] = (AA3[rn], np.array([float(line[30:38]), float(line[38:46]),
                                       float(line[46:54])], np.float32))
    return out


def p2rank_pocket(csv_path: str | Path, rank: int) -> list[tuple[str, int]]:
    """Residue keys of one P2Rank pocket, in the order P2Rank reports them."""
    for row in csv.DictReader(Path(csv_path).open()):
        row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
        if row.get("rank") and int(row["rank"]) == rank:
            keys = []
            for tok in row["residue_ids"].split():
                if "_" in tok:
                    c, n = tok.split("_", 1)
                    if n.lstrip("-").isdigit():
                        keys.append((c.strip() or "_", int(n)))
            return keys
    raise ValueError(f"pocket rank {rank} not found in {csv_path}")


def pocket_from_structure(pdb, prank_csv, rank: int, n_max: int = 64):
    """-> (pocket_seq, chain_seq, row indices of the pocket within the chain).

    The whole chain is returned because the language model sees the entire chain; only the
    pocket's rows are read back out of that pass.
    """
    ca = parse_ca(pdb)
    order = {k: i for i, k in enumerate(ca)}
    hits = [k for k in p2rank_pocket(prank_csv, rank) if k in ca][:n_max]
    if len(hits) < 2:
        raise ValueError("fewer than two pocket residues have a CA atom")
    chain_seq = "".join(a for a, _ in ca.values())
    return "".join(ca[k][0] for k in hits), chain_seq, [order[k] for k in hits]
