<div align="center">

# embeddingcache

[![PyPI](https://img.shields.io/pypi/v/:embeddingcache)(https://pypi.org/project/embeddingcache/)]
[![python](https://img.shields.io/badge/-Python_3.7_3.8_%7C_3.9_%7C_3.10%7C_3.11-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/) <br>
[![tests](https://github.com/turian/embeddingcache/actions/workflows/test.yml/badge.svg)](https://github.com/turian/embeddingcache/actions/workflows/test.yml)
[![license](https://img.shields.io/badge/License-Apache--2.0-green.svg?labelColor=gray)](https://github.com/turian/embeddingcache#license)

Retrieve text embeddings, but cache them locally if we have already computed them.

</div>

<br>

## Motivation

The use-case is if you are doing a handful of different NLP tasks
(or a single NLP pipeline you keep tuning) but don't want to recompute
embeddings.

## Quickstart

```
pip install embeddingcache
```

```
from embeddingcache.embeddingcache import get_embeddings
embeddings = get_embeddings(
            strs=["hi", "I love Berlin."],
            embedding_model="all-MiniLM-L6-v2",
            db_directory=Path("dbs/"),
            verbose=True,
        )
```

## Design assumptions

We use SQLite3 to cache embeddings. [This could be adapted easily,
since we use SQLAlchemy.]

We assume read-heavy loads, with one concurrent writer. (However,
we retry on write failures.)

We shard SQLite3 into two databases:
hashstring.db: hashstring table. Each row is a (unique, primary
key) SHA512 hash to text (also unique). Both fields are indexed.

[embedding_model_name].db: embedding table. Each row is a (unique,
primary key) SHA512 hash to a 1-dim numpy (float32) vector, which
we serialize to the table as bytes.

## Developer instructions

```
pre-commit install
pip install -e .
pytest
```

## TODO

* Update pyproject.toml
* Add tests
* Consider other hash functions?
* float32 and float64 support
* Consider adding optional joblib for caching?
* Different ways of computing embeddings (e.g. using an API) rather than locally
* S3 backup and/or
* WAL
* [LiteStream](https://fly.io/blog/all-in-on-sqlite-litestream/)
* Retry on write errors
* Other DB backends
* Best practices: Give specific OpenAI version number.
* RocksDB / RocksDB-cloud?
* Include model name in DB for sanity check on slugify.
* Validate on numpy array size.
* Validate BLOB size for hashes.
* Add optional libraries like openai and sentence-transformers
    * Also consider other embedding providers, e.g. cohere
    * And libs just for devs
* Consider the max_length of each text to embed, warn if we exceed
* pdoc3 and/or sphinx
* Normalize embeddings by default, but add option
* Option to return torch tensors
* Consider reusing the same DB connection instead of creating it
from scratch every time.
* Add batch_size parameter?
* Test check for collisions
* Use logging not verbose output.
* Rewrite using classes.
* Fix dependabot.
* Don't keep re-using DB session, store it in the class or global
* DRY.
* Suggest to use versioned OpenAI model
* Add device to sentence transformers
* Allow fast_sentence_transformers
* Test that things work if there are duplicate strings
* Remove DBs after test
* Do we have to have nested embedding.embedding for all calls?
* codecov and code quality shields
