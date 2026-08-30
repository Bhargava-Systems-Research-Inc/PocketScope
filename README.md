# PocketScope

Find the other proteins a drug might bind, by comparing binding pockets across the entire human
proteome.

PocketScope describes every pocket as a **tensor with one vector per lining residue**, built from
a frozen protein language model plus a small chemical descriptor, and retrieves similar pockets by
exact late-interaction search. The language model is used exactly as released.

One query is scored against all 153,805 human cavities in about 51 ms on a single GPU.

**Web server:** https://www.bhargavaresearch.org/pocketscope
**Prebuilt index and benchmarks:** https://doi.org/10.5281/zenodo.22178549

> Mohan K. and Bhargava Y. *Retrieval of binding sites across the AlphaFold human proteome using
> protein language model representations.*

---

## Installation

```bash
git clone https://github.com/Bhargava-Systems-Research-Inc/PocketScope
cd PocketScope
pip install -e .
```

That gives you the `pocketscope` command. It needs only `numpy` to search an index.
Add `pip install -e ".[gpu]"` for GPU search, or `".[encode]"` to also encode new pockets from
structure, which pulls in `torch` and the `esm` package.

## Obtaining the index

The prebuilt human pocketome — 153,805 cavities over 37,682 proteins — is on Zenodo at
https://doi.org/10.5281/zenodo.22178549. Download it and
point the tool at the folder:

```
pocket_index/
  pocket_tensor.npy        (153805, 64, 1165) float16, 22.9 GB
  pocket_index_rows.csv    row -> UniProt accession and P2Rank pocket rank
```

## Usage

**Inspecting an index**

```bash
pocketscope info --index pocket_index/
```

**Retrieving candidate off-targets**

```bash
pocketscope search --index pocket_index/ \
                   --pdb  AF-P24941-F1-model_v4.pdb \
                   --prank AF-P24941-F1-model_v4.pdb_predictions.csv \
                   --rank 1 --exclude P24941 --top-k 20
```

```
 rank  accession    pocket             score
    1  Q00534       Q00534_1           0.8418
    2  P11802       P11802_1           0.8133
    3  Q00526       Q00526_2           0.7887
```

`--pdb` is a structure, `--prank` is the P2Rank prediction CSV for it, and `--rank` picks which
detected pocket to use as the query. `--exclude` drops the query's own protein from the results.
Add `--csv hits.csv` to write the table instead of printing it, or `--device cpu` to search
without a GPU.

**Encoding and searching**

```bash
pocketscope encode --pdb protein.pdb --prank protein_predictions.csv --rank 1 --out query.npy
pocketscope search --index pocket_index/ --query query.npy
```

**Python interface**

```python
from pocketscope import PocketIndex, pocket_from_structure, encode_pocket
from pocketscope.encoder import embed_chain

pocket_seq, chain_seq, rows = pocket_from_structure("protein.pdb", "preds.csv", rank=1)
query = encode_pocket(pocket_seq, embed_chain(chain_seq)[rows])   # (L, 1165)

ix = PocketIndex("pocket_index/")
for hit in ix.search(query, top_k=10):
    print(hit["accession"], hit["score"])
```

## Repository contents

| file | what it does |
|---|---|
| `pocketscope/features.py` | the pocket representation, Equation 1 |
| `pocketscope/encoder.py` | frozen ESM-C 600M, chain in and per-residue vectors out |
| `pocketscope/structure.py` | reads PDB and P2Rank output into pocket residue sets |
| `pocketscope/index.py` | holds the index and runs the MaxSim search |
| `pocketscope/cli.py` | the `pocketscope` command |
| `examples/quickstart.py` | scores two pockets from the index |

Benchmarks and figure sources live in the Zenodo record rather than here, to keep this
repository to the implementation.

## Resources

- Web server — https://www.bhargavaresearch.org/pocketscope
- Prebuilt index, benchmarks and figure data — https://doi.org/10.5281/zenodo.22178549
- Source — https://github.com/Bhargava-Systems-Research-Inc/PocketScope

## Licence

MIT. See `LICENSE`.
