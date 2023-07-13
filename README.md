# embeddingcache

Retrieve embeddings, but cache them locally if we have already computed them.

The use-case is if you are doing a handful of different NLP tasks
(or a single NLP pipeline you keep tuning) but don't want to recompute
embeddings.

We use SQLite3 to cache embeddings. [This could be adapted easily,
since we use SQLAlchemy.]

## Spec

We use SQLite3 to cache embeddings. However, SQLAlchemy is used so
that the backend is portable to other to other backend DBs, possibly
even cloud DBs.

We assume read-heavy loads, with one concurrent writer. (However,
we retry on write failures.)

embeddingcache.py:


dbcache.py:

We shard SQLite3 into two databases:
hashstring.db: hashstring table. Each row is a (unique, primary
key) SHA512 hash to text (also unique). Both fields are indexed.

[embedding_model_name].db: embedding table. Each row is a (unique,
primary key) SHA512 hash to a 1-dim numpy (float32) vector, which
we serialize to the table as bytes.

## Developer instructions

```
pre-commit install
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
