"""
Database for storing hashids and their corresponding strings.

TODO: Consider converting this to a class?
TODO: Some of this code is reused with embeddingdb.py, consider refactoring.
"""

import hashlib
import sys
from pathlib import Path
from typing import List

from slugify import slugify
from sqlalchemy import Column, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
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
    """
    Get hashids for a list of strings.
    Also, cache the hashids in the database.

    Parameters
    ----------
    strs : List[str]
        List of strings to get hashids for.
    db_directory : Path
        Directory where the database is stored.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    List[bytes]
        List of hashids for the strings.
    """
    hashids = [
        compute_hashid(text)
        for text in tqdm(strs, disable=not verbose, desc="Computing hashids")
    ]
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

    TODO: Refactor into two functions?

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
    db_filepath = get_db_filename(
        db_directory=db_directory, db_basefilename="hashstring"
    )
    engine = create_engine(f"sqlite:///{db_filepath}")
    Session = sessionmaker(bind=engine)
    session = Session()

    idxs_of_missing_hashids = []
    # First, find all indexes of hashids that are NOT already in the database
    for i in tqdm(
        range(0, len(hashids), batch_size),
        disable=not verbose,
        desc="Finding missing hashids with in hashstring",
    ):
        batch_hashids = hashids[i : i + batch_size]
        existing_hashids = (
            session.query(HashString.hashid)
            .filter(HashString.hashid.in_(batch_hashids))
            .all()
        )
        existing_hashids = {x[0] for x in existing_hashids}
        idxs_of_missing_hashids.extend(
            i + j
            for j, hashid in enumerate(batch_hashids)
            if hashid not in existing_hashids
        )
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
        desc="Caching hashids to hashstring database",
    ):
        batch_hashids = [
            hashids[j] for j in idxs_of_missing_hashids[i : i + batch_size]
        ]
        batch_missing_strs = [
            strs[j] for j in idxs_of_missing_hashids[i : i + batch_size]
        ]
        new_hashstrings = [
            HashString(hashid=hashid, text=str)
            for hashid, str in zip(batch_hashids, batch_missing_strs)
        ]
        for hashid, str in zip(batch_hashids, batch_missing_strs):
            print(hashid, str)
        session.bulk_save_objects(new_hashstrings)
        # This is slower than one commit outside the loop, but is good
        # if we are concerned about memory overflow for many new hashstrings
        session.commit()


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
    db_filepath = db_directory / db_filename
    create_db_if_not_exists(db_filepath=db_filepath)
    return db_filepath


def create_db_if_not_exists(db_filepath: Path) -> None:
    """
    Create the database and table if it doesn't exist.

    Parameters
    ----------
    db_filepath : Path
        Full path to the database file.

    Returns
    -------
    None
    """

    if not db_filepath.exists():
        engine = create_engine(f"sqlite:///{db_filepath}")
        Base.metadata.create_all(bind=engine)
        # Is this stuff necessary?
        Session = sessionmaker(bind=engine)
        session = Session()
        session.commit()
        session.close()


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
