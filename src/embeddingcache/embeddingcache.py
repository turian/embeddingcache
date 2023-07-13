"""
"""

from pathlib import Path

from embeddingdb import get_embeddings


def get_embeddings(
    strs: List[str],
    embedding_model: str,
    db_directory: Path = Path("."),
    verbose: bool = False,
) -> np.ndarray:
    return get_embeddings(
        strs=strs,
        embedding_model=embedding_model,
        db_directory=db_directory,
        verbose=verbose,
    )
