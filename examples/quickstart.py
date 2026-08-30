"""Score two pockets against each other straight out of the index, no GPU required."""
import sys
import numpy as np
from pocketscope import PocketIndex

index_dir = sys.argv[1] if len(sys.argv) > 1 else "data/pocket_index"
ix = PocketIndex(index_dir, device="cpu")
print(f"{ix.n} pockets, {len(set(ix.accession))} proteins")

def pocket(i):
    row = np.asarray(ix.feats[i], np.float32)
    return row[np.abs(row).sum(1) > 0]          # drop the zero padding

a, b = pocket(0), pocket(1)
print(f"pocket 0 ({ix.pocket_id[0]}): {len(a)} residues")
print(f"pocket 1 ({ix.pocket_id[1]}): {len(b)} residues")
print(f"MaxSim(0 -> 1) = {(a @ b.T).max(1).mean():.4f}")
