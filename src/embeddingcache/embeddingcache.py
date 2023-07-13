"""
"""

import base64
import hashlib
import random
import re
import sqlite3
import sys
import time
import traceback
from datetime import datetime
from typing import List, Tuple

import faiss
import numpy as np
import openai
import tiktoken
from globals import get_device, memory

# from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


@memory.cache(ignore=["verbose"])
def get_embeddings(
    strs: List[str],
    model: str,
    dim: int,
    sqlite_path: str,
    verbose: bool = True,
) -> np.ndarray:
    return get_embeddings_uncached(
        strs, model, dim, sqlite_path=sqlite_path, verbose=verbose
    )


def get_embeddings_uncached(
    strs: List[str],
    model: str,
    dim: int,
    sqlite_path: str,
    verbose: bool = True,
) -> np.ndarray:
    table_name = model_to_table_name(model)
    # Create the table if it doesn't exist
    create_table_if_not_exists(table_name, sqlite_path=sqlite_path)
    # Call get_embeddings_batch on the strs
    embeddings = get_embeddings_batch_cache(
        strs, model, dim, sqlite_path=sqlite_path, verbose=verbose
    )
    assert embeddings.shape == (len(strs), dim)

    # Check the embeddings are normalized
    # Note that this might be very memory intensive, if the embedding matrix
    # is large. In that case, we could move it to work in batches.
    print("Checking that the embeddings are normalized...")
    n_rows = embeddings.shape[0]
    batch_size = 1024
    for i in tqdm(range(0, n_rows, batch_size), desc="Normalizing Embeddings"):
        batch_embeddings = embeddings[i : min(i + batch_size, n_rows), :]
        normalized_batch_embeddings = batch_embeddings / np.linalg.norm(
            batch_embeddings, axis=1, keepdims=True
        )

        assert np.allclose(
            normalized_batch_embeddings, batch_embeddings, atol=1e-6
        ), "Embeddings in batch {} are not normalized".format(i // batch_size)

        if i + batch_size < n_rows:
            del normalized_batch_embeddings

    return embeddings


def create_table_if_not_exists(table_name: str, sqlite_path: str):
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    # Create table with the given schema if it doesn't exist
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            str TEXT PRIMARY KEY,
            embedding BLOB NOT NULL
        );
    """
    )


def compute_and_cache_missing_embeddings(
    strs: List[str],
    model: str,
    dim: int,
    batch_size: int,
    sqlite_path: str,
    verbose: bool = True,
):
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()
    table_name = model_to_table_name(model)

    # Batch strs into batch_size for handling large strs
    strs_batches = [strs[i : i + batch_size] for i in range(0, len(strs), batch_size)]

    found_strs = set()
    from tqdm.auto import tqdm

    if not verbose:
        tqdm = lambda x: x
    for strs_batch in tqdm(strs_batches):
        # Prepare the query for each batch
        query = f"SELECT str FROM {table_name} WHERE str IN ({','.join('?' * len(strs_batch))})"

        # Execute the query
        cursor.execute(query, strs_batch)

        found_strs.update([s for s, in cursor.fetchall()])

    missing_strs = [s for s in strs if s not in found_strs]

    if verbose:
        print(f"Found {len(found_strs)} out of {len(strs)} strings")
        print(f"Missing {len(missing_strs)} out of {len(strs)} strings")

    if missing_strs:
        # Shuffle the missing_strs list
        # Do this in a random order so that if the program crashes, we can
        # still make progress
        random.shuffle(missing_strs)

        if model != "text-embedding-ada-002":
            # Batch missing_strs into batch_size
            missing_batches = [
                missing_strs[i : i + batch_size]
                for i in range(0, len(missing_strs), batch_size)
            ]
        else:
            enc = tiktoken.get_encoding("cl100k_base")
            missing_batches = []
            missing_batch = []
            missing_batch_tokens = 0
            for missing_str in tqdm(missing_strs):
                if len(enc.encode(missing_str)) + missing_batch_tokens >= 8191:
                    missing_batches.append(missing_batch)
                    missing_batch = []
                    missing_batch_tokens = 0
                missing_batch.append(missing_str)
                missing_batch_tokens += len(enc.encode(missing_str))
            if missing_batch:
                missing_batches.append(missing_batch)

        for missing_batch in tqdm(missing_batches):
            try:
                # Compute embeddings for missing strings
                computed_embeddings = compute_embeddings_batch(
                    missing_batch, model, dim
                )
            except Exception as e:
                print(f"Error {e} in compute_embeddings_batch", file=sys.stderr)
                continue

            try:
                # Insert computed embeddings into the database
                insert_query = (
                    f"INSERT INTO {table_name} (str, embedding) VALUES (?, ?)"
                )
                for s, emb in zip(missing_batch, computed_embeddings):
                    # Convert embeddings to blob
                    emb_blob = sqlite3.Binary(emb.astype("float32").tobytes())
                    cursor.execute(insert_query, (s, emb_blob))

                # Commit the changes
                conn.commit()
            except Exception as e:
                # Get the current system time
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Print traceback, system time, input arguments, and the failed query
                print(
                    f"Error {e} in get_embeddings_batch at {current_time}",
                    file=sys.stderr,
                )
                print("Missing batch:", missing_batch, file=sys.stderr)
                print(f"Failed query: {insert_query}", file=sys.stderr)
                traceback.print_exception(*sys.exc_info(), file=sys.stderr)
                continue

        # vacuum_db(sqlite_path)

    # Close the connection
    conn.close()


def get_embeddings_batch_cache(
    # strs: List[str], model: str, dim: int, batch_size: int = 32
    strs: List[str],
    model: str,
    dim: int,
    sqlite_path: str,
    batch_size: int = 1024,
    verbose: bool = True,
) -> np.ndarray:
    compute_and_cache_missing_embeddings(
        strs, model, dim, batch_size, verbose=verbose, sqlite_path=sqlite_path
    )

    embeddings = {}

    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()
    table_name = model_to_table_name(model)

    # Batch strs into batch_size for handling large strs
    strs_batches = [strs[i : i + batch_size] for i in range(0, len(strs), batch_size)]

    found_strs = set()
    from tqdm.auto import tqdm

    if not verbose:
        tqdm = lambda x: x
    for strs_batch in tqdm(strs_batches):
        # Prepare the query for each batch
        query = f"SELECT str, embedding FROM {table_name} WHERE str IN ({','.join('?' * len(strs_batch))})"

        # Execute the query
        cursor.execute(query, strs_batch)

        this_embeddings = dict(cursor.fetchall())
        for s, emb in this_embeddings.items():
            embeddings[s] = np.frombuffer(emb, dtype=np.float32)

    # assert len(embeddings) == len(strs), f"{len(embeddings)} != {len(strs)}"
    assert set(embeddings.keys()) == set(strs)
    for s in strs:
        assert embeddings[s].shape == (dim,)
    ordered_embeddings = np.vstack([embeddings[s] for s in strs])
    ## TODO: remove this
    # print(f"oembeddings shape:", ordered_embeddings.shape)
    # print(f"oembeddings std:", ordered_embeddings.std(axis=0).mean())
    # print(f"oembeddings std:", ordered_embeddings.std(axis=1).mean())
    assert ordered_embeddings.shape == (len(strs), dim)
    return ordered_embeddings


def vacuum_db(sqlite_path: str):
    print("Vacuuming database...")
    conn = sqlite3.connect(sqlite_path)
    conn.execute("VACUUM")
    conn.close()
    print("...done vacuuming database")


def list_all_tables(sqlite_path: str) -> List[str]:
    # Connect to the SQLite database
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    # Query to list all tables in the database
    query = "SELECT name FROM sqlite_master WHERE type='table';"

    # Execute the query and fetch the results
    cursor.execute(query)
    table_names = cursor.fetchall()

    # Close the connection
    conn.close()

    # Return table names as a list of strings
    return [table_name for (table_name,) in table_names]


def model_to_table_name(model: str) -> str:
    """
    Convert a model name to a SQLite3 table name.
    We use a hash of the model name to ensure that the table name is valid.
    """
    # Create a hash using hashlib's sha256
    model_hash = hashlib.sha256(model.encode("utf-8")).digest()

    # Encode the hash as a base64 string
    table_name_base64 = base64.b64encode(model_hash).decode("utf-8")

    # Replace all non-alphanumeric and non-underscore characters with an alphabetic string
    # table_name = re.sub("[^a-zA-Z0-9_]", "_", table_name_base64)
    table_name = re.sub("[^a-zA-Z0-9_]", "zz", table_name_base64)

    # Ensure the table name starts with a letter by adding a prefix
    return f"t_{table_name}"


embedding_models = {}


"""
def get_embedding_model(model_name: str) -> SentenceTransformer:
    device = get_device()
    if (model_name, device) not in embedding_models:
        embedding_models[(model_name, device)] = SentenceTransformer(
            model_name, device=device
        )
    return embedding_models[(model_name, device)]
"""


def compute_embeddings_batch(strs: List[str], model: str, dim: int) -> np.ndarray:
    """
    Compute embeddings for a batch of strings.

    Normalize it at the end.
    """
    if model != "text-embedding-ada-002":
        device = get_device()
        model = get_embedding_model(model)
        embeddings = (
            model.encode(strs, convert_to_tensor=True, device=device).cpu().numpy()
        )
    else:
        assert model == "text-embedding-ada-002"
        # Compute tokens for each string
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = sum([len(enc.encode(s)) for s in strs])
        assert tokens < 8191, f"Too many tokens: {tokens}"
        r = openai.Embedding.create(input=strs, model="text-embedding-ada-002")
        assert "data" in r
        embeddings = np.array([e["embedding"] for e in r["data"]])
    assert embeddings.shape == (
        len(strs),
        dim,
    ), f"{embeddings.shape} != {(len(strs), dim)}"
    # Normalize the embeddings
    # We use faiss because it mutates the array in-place
    faiss.normalize_L2(embeddings.astype(np.float32))
    return embeddings


@memory.cache(ignore=["batch_size"])
def get_distances_embeddings(
    strpairs: List[Tuple[str, str]],
    model: str,
    dim: int,
    sqlite_path: str,
    batch_size: int = 1024 // 2,
) -> np.ndarray:
    strpairs_batches = [
        strpairs[i : i + batch_size] for i in range(0, len(strpairs), batch_size)
    ]

    distances = []

    print(len(strpairs_batches))
    for strpairs_batch in tqdm(strpairs_batches):
        # Flatten the strpairs_batch to a list of unique strings
        unique_strs = list(set([s for pair in strpairs_batch for s in pair]))

        # Get embeddings of unique strings
        # Don't cache results of this call
        normalized_embeddings = get_embeddings(
            unique_strs,
            model,
            dim,
            sqlite_path=sqlite_path,
            verbose=False,
        )

        # They are pre-normalized
        """
        # Normalize the embeddings
        normalized_embeddings = embeddings / np.linalg.norm(
            embeddings, axis=1, keepdims=True
        )
        """

        # Create a mapping of string to its normalized embedding
        str_to_embedding = dict(zip(unique_strs, normalized_embeddings))

        # Compute euclidean distance for each pair in the batch
        batch_distances = [
            np.linalg.norm(str_to_embedding[s1] - str_to_embedding[s2])
            for s1, s2 in strpairs_batch
        ]
        distances.extend(batch_distances)

    # Return distances as an ndarray
    distances = np.array(distances)
    assert distances.shape == (len(strpairs),)
    return distances
