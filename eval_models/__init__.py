"""Evaluation helpers for face-swap model outputs.

This package adapts heterogeneous output folders into the simple directory
layout expected by REFace's ``eval_tool`` scripts.
"""

from .pairs import EvalPair, PairImageReader, read_pairs_csv
from .resolvers import CandidateImage, detect_resolvers

__all__ = [
    "CandidateImage",
    "EvalPair",
    "PairImageReader",
    "detect_resolvers",
    "read_pairs_csv",
]
