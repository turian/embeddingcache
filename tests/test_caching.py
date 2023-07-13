# the inclusion of the tests module is not meant to offer best practices for
# testing in general, but rather to support the `find_packages` example in
# setup.py that excludes installing the "tests" package

import unittest
from pathlib import Path

import numpy as np

from embeddingcache.embeddingcache import get_embeddings


class TestCaching(unittest.TestCase):
    def test_add_one(self):
        perm = [2, 0, 1]
        strs = ["hello world", "I love berlin", "I hate haters"]
        strs2 = [strs[i] for i in perm]
        e1 = get_embeddings(
            strs=strs,
            embedding_model="all-MiniLM-L6-v2",
            db_directory=Path("."),
            verbose=True,
        )
        # TODO: Way to test that caching actually worked?
        e2 = get_embeddings(
            strs=strs2,
            embedding_model="all-MiniLM-L6-v2",
            db_directory=Path("."),
            verbose=True,
        )

        # Check that the embeddings are the same, when re-ordered
        self.assertTrue(np.allclose(e1[perm], e2))


if __name__ == "__main__":
    unittest.main()
