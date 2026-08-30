"""Command line interface: pocketscope <command>."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np


def _encode(a):
    from .structure import pocket_from_structure
    from .encoder import embed_chain
    from .features import encode_pocket, N_MAX
    pocket_seq, chain_seq, rows = pocket_from_structure(a.pdb, a.prank, a.rank, a.n_max or N_MAX)
    print(f"pocket rank {a.rank}: {len(pocket_seq)} residues from a {len(chain_seq)}-residue chain",
          file=sys.stderr)
    esm = embed_chain(chain_seq, a.device)[rows]
    return encode_pocket(pocket_seq, esm, a.alpha)


def cmd_encode(a):
    q = _encode(a)
    np.save(a.out, q)
    print(f"wrote {a.out}  {q.shape}")


def cmd_search(a):
    from .index import PocketIndex
    q = np.load(a.query) if a.query else _encode(a)
    ix = PocketIndex(a.index, device=a.device)
    print(f"index: {ix.n} pockets x {ix.n_max} residues x {ix.dim} features", file=sys.stderr)
    hits = ix.search(q, top_k=a.top_k, exclude=a.exclude)
    if a.csv:
        w = csv.DictWriter(open(a.csv, "w", newline=""),
                           fieldnames=["rank", "pocket_id", "accession", "score"])
        w.writeheader(); w.writerows(hits)
        print(f"wrote {a.csv}")
    else:
        print(f"{'rank':>5}  {'accession':<12} {'pocket':<18} score")
        for h in hits:
            print(f"{h['rank']:>5}  {h['accession']:<12} {h['pocket_id']:<18} {h['score']:.4f}")


def cmd_info(a):
    from .index import PocketIndex
    ix = PocketIndex(a.index, device="cpu")
    L = ix.lengths()
    print(f"pockets      {ix.n}")
    print(f"proteins     {len(set(ix.accession))}")
    print(f"shape        ({ix.n}, {ix.n_max}, {ix.dim}) {ix.feats.dtype}")
    print(f"residues     mean {L[L>0].mean():.2f}  median {int(np.median(L[L>0]))}  max {L.max()}")
    print(f"size on disk {ix.feats.nbytes/1e9:.1f} GB")


def main(argv=None):
    p = argparse.ArgumentParser(prog="pocketscope", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_encode_args(s):
        s.add_argument("--pdb", required=True, help="structure file")
        s.add_argument("--prank", required=True, help="P2Rank predictions CSV for that structure")
        s.add_argument("--rank", type=int, default=1, help="which P2Rank pocket (default 1)")
        s.add_argument("--alpha", type=float, default=0.5, help="physicochemical block weight")
        s.add_argument("--n-max", type=int, dest="n_max", default=None)
        s.add_argument("--device", default="cuda")

    e = sub.add_parser("encode", help="turn one pocket into its (L, 1165) tensor")
    add_encode_args(e); e.add_argument("--out", required=True); e.set_defaults(f=cmd_encode)

    s = sub.add_parser("search", help="rank the index against one pocket")
    s.add_argument("--index", required=True, help="directory holding pocket_tensor.npy")
    s.add_argument("--query", help="a .npy from `encode`; omit to encode on the fly")
    s.add_argument("--pdb"); s.add_argument("--prank"); s.add_argument("--rank", type=int, default=1)
    s.add_argument("--alpha", type=float, default=0.5)
    s.add_argument("--n-max", type=int, dest="n_max", default=None)
    s.add_argument("--top-k", type=int, dest="top_k", default=20)
    s.add_argument("--exclude", help="UniProt accession to drop from the results, e.g. the query")
    s.add_argument("--csv", help="write results here instead of printing")
    s.add_argument("--device", default="cuda")
    s.set_defaults(f=cmd_search)

    i = sub.add_parser("info", help="summarise an index")
    i.add_argument("--index", required=True); i.set_defaults(f=cmd_info)

    a = p.parse_args(argv)
    if a.cmd == "search" and not a.query and not (a.pdb and a.prank):
        p.error("search needs either --query or both --pdb and --prank")
    return a.f(a)


if __name__ == "__main__":
    main()
