"""
WRITEME
"""


from pathlib import Path
from typing import List

import hashstringdb
import numpy as np
from sqlalchemy import Column, LargeBinary, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from tqdm.auto import tqdm

Base = declarative_base()


# Embedding Table
class Embedding(Base):
    __tablename__ = "embedding"

    hashid = Column(String(128), primary_key=True)
    embedding = Column(LargeBinary)


"""
def create_db_and_table(embedding_model_name):  
    engine = create_engine(f'sqlite:////full/path/to/{embedding_model_name}.db')
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)

embedding_model_name="your_model_name"
db_session = create_db_and_table(embedding_model_name)
session = db_session()

new_embedding = Embedding(sha512_hash='sample_hash', numpy_vector=b'sample_numpy_vector')
session.add(new_embedding)

session.commit()
"""


def get_embeddings(
    strs: List[str],
    embedding_model: str,
    db_directory: Path = Path("."),
    verbose: bool = False,
) -> np.ndarray:
    # Make sure all hashids are in the hashstring database
    hashids = hashstringdb.get_hashids(strs, db_directory=db_directory, verbose=verbose)
    embeddings = get_embeddings_with_caching(
        strs=strs,
        hashids=hashids,
        embedding_model=embedding_model,
        db_directory=db_directory,
        verbose=verbose,
    )
    assert len(embeddings) == len(hashids)
    return embeddings


def get_embeddings_with_caching(
    strs: List[str],
    hashids: List[bytes],
    embedding_model: str,
    db_directory: Path = Path("."),
    verbose: bool = False,
    batch_size: int = 1024,
) -> np.ndarray:
    db_filename = get_db_filename(
        db_directory=db_directory, db_basename=f"embedding_{embedding_model}"
    )
    engine = create_engine(f"sqlite:///{db_filename}")
    Session = sessionmaker(bind=engine)
    session = Session()

    idxs_of_missing_hashids = []
    # First, find all indexes of hashids that are NOT already in the embedding database
    for i in tqdm(
        range(0, len(hashids), batch_size),
        disable=not verbose,
        desc="Finding missing hashids with in embeddings db",
    ):
        batch_hashids = hashids[i : i + batch_size]
        existing_hashids = (
            session.query(Embedding.hashid)
            .filter(Embedding.hashid.in_(batch_hashids))
            .all()
        )
        existing_hashids = {x[0] for x in existing_hashids}
        idxs_of_missing_hashids.extend(
            i + j
            for j, hashid in enumerate(batch_hashids)
            if hashid not in existing_hashids
        )

    """"
    Now, compute and add all embeddings for the missing strings.

    This solution assumes the process has enough memory to hold all
    new `Embedding` instances in a batch in memory at once. If
    this is not the case, further adaptations would be needed, such
    as flush and clear the session after each commit with
    `session.flush()` and `session.expunge_all()`
    """
    for i in tqdm(
        range(0, len(idxs_of_missing_hashids), batch_size),
        disable=not verbose,
        desc="Computing new embeddings",
    ):
        batch_hashids = idxs_of_missing_hashids[i : i + batch_size]
        batch_missing_strs = [strs[i] for i in batch_hashids]
        batch_embeddings = embedding_model.encode(batch_missing_strs)
        new_hashstrings = [
            Embedding(hashid=hashid, embeding=embedding)
            for hashid, embedding in zip(batch_hashids, batch_embeddings.tobytes())
            # for hashid, embedding in zip(batch_hashids, batch_embeddings.astype("float32").tobytes())
        ]
        session.bulk_save_objects(new_hashstrings)
        # This is slower than one commit outside the loop, but is good
        # if we are concerned about memory overflow for many new hashstrings
        session.commit()

    # Now, get all embeddings
    # Might be faster not to do DB lookups several times
    embeddings = (
        session.query(Embedding.numpy_vector)
        .filter(Embedding.hashid.in_(hashids))
        .all()
    )
    # Put the embeddings in the same order as the hashids
    embeddings = {x[0]: x[1] for x in embeddings}
    embeddings = [np.frombuffer(embeddings[hashid]) for hashid in hashids]
    embeddings = np.vstack(embeddings)
    assert len(embeddings) == len(hashids)
    if verbose:
        print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


##### The rest of this is mainly copy-pasta from hashstringdb.py


def get_db_filename(db_directory: Path, db_basefilename: str) -> Path:
    """
    Get the full path to the database file.

    Create the directory if it doesn't exist.

    Parameters
    ----------
    db_directory : Path
        Directory where the database is stored.
    db_filename : str
        Name of the database file.
        Default: "hashstring.sqlite"

    Returns
    -------
    Path
        Full path to the database file.
    """
    # TODO: Make sure there are no collisions of slugified names
    db_filename = f"{slugify(db_basefilename)}.sqlite"
    db_directory.mkdir(parents=True, exist_ok=True)
    return db_directory / db_filename


def create_db_if_not_exists(
    db_directory: Path, db_filename: str = "hashstring.sqlite"
) -> None:
    """ """
    db_filename = get_db_filename(db_directory=db_directory, db_filename=db_filename)
    # Create DB + table if it doesn't exist
    if not db_filename.exists():
        engine = create_engine(f"sqlite:///{db_filename}")
        Base.metadata.create_all(bind=engine)
