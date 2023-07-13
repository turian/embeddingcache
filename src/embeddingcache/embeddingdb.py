"""
WRITEME
"""


from pathlib import Path
from typing import List

import hashstringdb
from sqlalchemy import Column, LargeBinary, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from tqdm.auto import tqdm

Base = declarative_base()


# Embedding Table
class Embedding(Base):
    __tablename__ = "embedding"

    hashid = Column(String(128), primary_key=True)
    numpy_vector = Column(LargeBinary)


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
    embeddings = get_embeddings_from_hashids(
        hashids=hashstringdb.get_hashids(
            strs, db_directory=db_directory, verbose=verbose
        ),
        embedding_model=embedding_model,
        db_directory=db_directory,
        verbose=verbose,
    )
    assert len(embeddings) == len(hashids)
    return embeddings


def get_embeddings_from_hashids(
    hashids: List[bytes],
    embedding_model: str,
    db_directory: Path = Path("."),
    verbose: bool = False,
) -> np.ndarray:
    return get_embeddings_with_caching(
        hashids=hashids,
        embedding_model=embedding_model,
        db_directory=db_directory,
        verbose=verbose,
    )


# TODO: A lot of this code can be reused from hashstringdb.py
def get_embeddings_with_caching(
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
    # First, find all indexes of hashids that are NOT already in the database
    for i in tqdm(
        range(0, len(hashids), batch_size),
        disable=not verbose,
        desc="Finding missing hashids in embedding db",
    ):
        batch_missing_hashids = hashids[i : i + batch_size]
        batch_massing_strs = strs[i : i + batch_size]
        existing_hashids = (
            session.query(HashString.hashid)
            .filter(HashString.hashid.in_(batch_missing_hashids))
            .all()
        )
        existing_hashids = {x[0] for x in existing_hashids}
        idxs_of_missing_hashids.extend(
            i + j
            for j, hashid in enumerate(batch_missing_hashids)
            if hashid not in existing_hashids
        )
    if verbose:
        print(
            f"Found {len(idxs_of_missing_hashids)} hashids not in the database, {len(hashids) - len(idxs_of_missing_hashids)} were cached.",
            file=sys.stderr,
        )


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
    db_filename = slugify(db_basefilename) + ".sqlite"
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


def compute_hashid(text: str) -> bytes:
    """
    Compute the hashid of a string.

    We use SHA512.

    Parameters
    ----------
    text : str
        String to hash.

    Returns
    -------
    bytes
        Hashid of the string.
    """
    return hashlib.sha512(text.encode("utf-8")).digest()
