from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


def embed(
    strs: List[str], embedding_model: str, verbose: bool, normalize: bool = True
) -> np.ndarray:
    """
    Compute embeddings for a batch of strings.

    Note that sentence transformers has a batch_size parameter, so we ignore our batch.

    TODO: Add device
    """
    if embedding_model == "text-embedding-ada-002":
        raise ValueError(f"Unknown model: {model}")
    # TODO: Would be faster to cache this
    model = SentenceTransformer(embedding_model)
    embeddings = model.encode(
        strs,
        convert_to_numpy=True,
        show_progress_bar=verbose,
        normalize_embeddings=normalize,
    )
    """
    else:
        assert model == "text-embedding-ada-002"
        # Compute tokens for each string
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = sum([len(enc.encode(s)) for s in strs])
        assert tokens < 8191, f"Too many tokens: {tokens}"
        r = openai.Embedding.create(input=strs, model="text-embedding-ada-002")
        assert "data" in r
        embeddings = np.array([e["embedding"] for e in r["data"]])

        if normalize:
            embeddings = sklearn.preprocessing.normalize(embeddings)
    """
    assert len(embeddings) == len(strs)
    return embeddings
