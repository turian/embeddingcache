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
    """ "
    Get embeddings for a list of strings.
    We retrieve from the database if possible, otherwise we compute and cache.

    Parameters
    ----------
    strs : List[str]
        List of strings to get embeddings for.
    embedding_model : str
        Name of the embedding model to use.
    db_directory : Path
        Directory where the database is stored.
    verbose : bool
        Whether to print progress bars.

    Returns
    -------
    np.ndarray
        Embeddings for the list of strings.
    """
    return embeddingdb.get_embeddings(
        strs=strs,
        embedding_model=embedding_model,
        db_directory=db_directory,
        verbose=verbose,
    )
