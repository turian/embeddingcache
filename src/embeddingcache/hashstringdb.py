"""
WRITEME

TODO: Consider converting this to a class?
"""

import hashlib
from pathlib import Path
from typing import List

from sqlalchemy import Column, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from tqdm.auto import tqdm

Base = declarative_base()


# Hashstring Table
class HashString(Base):
    __tablename__ = "hashstring"

    hashid = Column(String(128), primary_key=True)
    text = Column(String, unique=True)


"""
def create_db_and_table():  
    engine = create_engine('sqlite:////full/path/to/hashstring.db')
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)

db_session = create_db_and_table()
session = db_session()

new_hashstring = HashString(sha512_hash='sample_hash', text='sample_text')
session.add(new_hashstring)

session.commit()
"""


def get_hashids(
    strs: List[str],
    db_directory: Path = Path("."),
    verbose: bool = False,
) -> List[bytes]:
    hashids = []
    for i in tqdm(strs, disable=not verbose, desc="Computing hashids"):
        hashids.append(compute_hashid(str))
    assert len(hashids) == len(strs)
    cache_hashid_to_strs(
        hashids=hashids, strs=strs, db_directory=db_directory, verbose=verbose
    )
    return hashids


def cache_hashid_to_strs(
    hashids: List[bytes],
    strs: List[str],
    db_directory: Path,
    verbose: bool = False,
    batch_size: int = 1024,
):
    """
    Find hashids that are already in the database. For those that
    are not, add them to the database with their string.

    Operate in batch writes of size `batch_size`.

    Parameters
    ----------
    hashids : List[bytes]
        List of hashids to check against the database.
    strs : List[str]
        List of strings to check against the database.
    db_directory : Path
        Directory where the database is stored.
    verbose : bool
        Whether to print progress.
    batch_size : int
        Number of hashids to check at a time.
        Default: 1024

    Returns
    -------
    None
    """
    assert len(hashids) == len(strs)
    db_filename = get_db_filename(db_directory=db_directory)
    engine = create_engine(f"sqlite:///{db_filename}")
    Session = sessionmaker(bind=engine)
    session = Session()

    idxs_of_missing_hashids = []
    # First, find all indexes of hashids that are NOT already in the database
    for i in tqdm(
        range(0, len(hashids), batch_size), disable=not verbose, desc="Caching hashids"
    ):
        batch_missing_hashids = hashids[i : i + batch_size]
        batch_massing_strs = strs[i : i + batch_size]
        existing_hashids = (
            session.query(HashString.hashid)
            .filter(HashString.hashid.in_(batch_missing_hashids))
            .all()
        )
        existing_hashids = set([x[0] for x in existing_hashids])
        for j, hashid in enumerate(batch_missing_hashids):
            if hashid not in existing_hashids:
                idxs_of_missing_hashids.append(i + j)

    if verbose:
        print(
            f"Found {len(idxs_of_missing_hashids)} hashids not in the database, {len(hashids) - len(idxs_of_missing_hashids)} were cached.",
            file=sys.stderr,
        )

    """"
    Now, add all missing hashids to the database, in batches.

    This solution assumes the process has enough memory to hold all
    new `HashString` instances in a batch in memory at once. If
    this is not the case, further adaptations would be needed, such
    as flush and clear the session after each commit with
    `session.flush()` and `session.expunge_all()`
    """
    for i in tqdm(
        range(0, len(idxs_of_missing_hashids), batch_size),
        disable=not verbose,
        desc="Caching hashids",
    ):
        batch_missing_hashids = idxs_of_missing_hashids[i : i + batch_size]
        batch_missing_strs = [strs[i] for i in batch_missing_hashids]
        new_hashstrings = [
            HashString(hashid=hashid, text=str)
            for hashid, str in zip(batch_missing_hashids, batch_missing_strs)
        ]
        session.bulk_save_objects(new_hashstrings)
        # This is slower than one commit outside the loop, but is good
        # if we are concerned about memory overflow for many new hashstrings
        session.commit()


def get_db_filename(db_directory: Path, db_filename: str = "hashstring.sqlite") -> Path:
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
