"""
"""

from pathlib import Path
from typing import List

import numpy as np

import embeddingcache.embeddingdb as embeddingdb


def get_embeddings(
    strs: List[str],
    embedding_model: str,
    db_directory: Path = Path("."),
    verbose: bool = False,
) -> np.ndarray:
    return embeddingdb.get_embeddings(
        strs=strs,
        embedding_model=embedding_model,
        db_directory=db_directory,
        verbose=verbose,
    )
