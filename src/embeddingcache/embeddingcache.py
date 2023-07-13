"""
"""

from pathlib import Path
from typing import List

import embeddingdb
import hashstringdb
import numpy as np
from tqdm.auto import tqdm


def get_embeddings(
    strs: List[str],
    embedding_model: str,
    db_directory: Path = Path("."),
    verbose: bool = False,
) -> np.ndarray:
    hashids = hashstringdb.get_hashids(strs, db_directory=db_directory, verbose=verbose)
