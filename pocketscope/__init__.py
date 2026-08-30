"""PocketScope: proteome-scale binding-site retrieval with frozen protein language models."""
from .features import encode_pocket, match_feature, residue_phys, ALPHA, FEAT_DIM, N_MAX
from .index import PocketIndex
from .structure import pocket_from_structure, parse_ca, p2rank_pocket

__version__ = "1.0.0"
__all__ = ["encode_pocket", "match_feature", "residue_phys", "PocketIndex",
           "pocket_from_structure", "parse_ca", "p2rank_pocket",
           "ALPHA", "FEAT_DIM", "N_MAX"]
