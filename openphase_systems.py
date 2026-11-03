"""Load the OpenPhase condition-level records by system.

The tables live in ``data/<system>/records.tsv`` (see ``data/README.md`` for the columns):

    protein        protein only                           7,490 records
    protein_RNA    protein + RNA                          2,511 records
    protein_DNA    protein + DNA                            662 records
    protein_csat   protein, saturation concentration        776 rows

The archived loaders ``zhanglab/db1.py`` (LLPSDB v2), ``zhanglab/db2.py`` (laboratory notebooks) and
``zhanglab/db3.py`` (RNAPSEC) build one embedded dataset per *source*; this module reads the merged records
per *system*.
"""
import os.path as osp
from typing import Dict, Iterable, Optional

import pandas as pd

DATA_DIR = osp.join(osp.dirname(osp.abspath(__file__)), "data")

SYSTEMS = {
    "protein": "protein only",
    "protein_RNA": "protein + RNA",
    "protein_DNA": "protein + DNA",
    "protein_csat": "protein, saturation concentration",
}


def list_systems() -> pd.DataFrame:
    """One row per system: records, sequences, UniProt accessions, papers, sources, labels."""
    return pd.read_csv(osp.join(DATA_DIR, "systems.tsv"), sep="\t")


def load_system(system: str, source: Optional[str] = None) -> pd.DataFrame:
    """Records of one system as a data frame; optionally keep one source (e.g. 'RNAPSEC')."""
    if system not in SYSTEMS:
        raise ValueError(f"unknown system {system!r}; choose from {sorted(SYSTEMS)}")
    df = pd.read_csv(osp.join(DATA_DIR, system, "records.tsv"), sep="\t", low_memory=False)
    df.insert(1, "system", system)
    if source is not None:
        df = df[df["source"].str.startswith(source)]
    return df.reset_index(drop=True)


def load_all(systems: Optional[Iterable[str]] = None) -> Dict[str, pd.DataFrame]:
    """Dictionary system -> records."""
    return {s: load_system(s) for s in (systems or SYSTEMS)}


if __name__ == "__main__":
    print(list_systems().to_string(index=False))
