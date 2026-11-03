# zhanglab: loaders of the three original sources

Archive of the first OpenPhase loaders, one per source:

| file | source |
|---|---|
| `db1.py` | LLPSDB v2 (downloaded from the LLPSDB site) |
| `db2.py` | records collected from laboratory notebooks (Zhang laboratory) |
| `db3.py` | RNAPSEC |

Each loader downloads its raw data, parses it to tables and embeds it as AnnData (`.h5ad`).
They import `meta_config.py` and `meta_embed.py` from the repository root, so run them from the root, for example `from zhanglab.db1 import DB1Dataset`.
The records by system are in [`../data/`](../data/README.md); they do not contain the records of `db2`.
