# OpenPhase records by system

Condition-level records of single-protein phase-separation experiments, one folder per system.
A record is one experimental condition of one system with its outcome: a binary label (phase separation or not) or, for the saturation-concentration system, a measured value.

| folder | system | records | sequences | UniProt | papers | positive / negative |
|---|---|---:|---:|---:|---:|---|
| `protein/` | protein only | 7,490 | 1,163 | 311 | 364 | 4,416 / 3,074 |
| `protein_RNA/` | protein + RNA | 2,511 | 241 | 85 | 125 | 1,751 / 760 |
| `protein_DNA/` | protein + DNA | 662 | 99 | 42 | 41 | 446 / 216 |
| `protein_csat/` | protein, saturation concentration | 776 | 42 | 3 | 2 | no binary label |

`systems.tsv` has the same numbers and the records per source. In the three binary systems a nucleic acid, if any, is one of the conditions (`na_*` columns). The system of a record is the set of its components without concentrations.

```python
from openphase_systems import list_systems, load_system
list_systems()
rna = load_system("protein_RNA")            # or load_system("protein_RNA", source="RNAPSEC")
```

## Sources

LLPSDB v2 (download files, rows listed only on its website, digitised phase-diagram tables), LLPSense, RNAPSEC, a pilot extraction from PhaSepDB 3 papers by language-model agents (`PhaSepDB3_pilot_LLM`), and, for the saturation-concentration system, Bremer 2022 and Ausserwöger 2025.
`db1.py`, `db2.py` and `db3.py` are the loaders of the three original sources (LLPSDB v2, laboratory notebooks, RNAPSEC); they download from the upstream sources and build one embedded dataset per source.
The records of the laboratory-notebook source (db2) are **not** in these tables.

## Columns

| group | columns |
|---|---|
| identity | `record_id`, `system`, `source`, `paper` (PMID or DOI), `uniprot`, `protein_name`, `seq_len`, `sequence_state` (how the construct sequence was resolved), `system_id` (hash of the components), `mutation`, `fusion`, `cleaved`, `ptm` |
| outcome | `label` (1 = phase separation, 0 = none; empty for saturation-concentration rows), `morphology`, `phase_state`; saturation concentration: `readout_class` (`dilute`, `csat_boundary`, `dilute_linescan`, `cloud_point`, `dense`), `log10_value_uM`, `scan_axis` |
| groups for splitting | `cluster_30`, `cluster_20` (MMseqs2 clusters at 30 % and 20 % identity, coverage 80 %) |
| conditions | `log10_protein_conc_uM`, `pH`, `temp_C`, `MgCl2_mM`, `NaCl_mM`, `KCl_mM`, `ionic_strength_mM`, crowders `PEG1k`, `PEG6k`, `PEG20k`, `Ficoll`, `Dex40`, `Dex70`, `Glycerol` (per cent as reported) |
| buffer and additives | `buf_*` (buffer families), `add_*` (additive classes), 0/1 flags |
| missing-value state | `st_*` next to each condition: `obs` observed, `miss` missing, `zero` reported as absent, `rt` room temperature only (`temp_C` = 22), `none` / `other` for buffer and additive |
| nucleic acid | `na_class` (`none`, `RNA`, `DNA`), `na_len_log10`, `na_conc_log10`, `na_to_protein_ratio_log10`, `na_charge_ratio_log10`, `na_gc`, `na_purine`, `na_maxrun`, `na_g4`, `na_mfe_per_nt`, `na_paired_frac`, `na_len_source`, `na_homopolymer`, `na_name_*` (name classes), `na_has_sequence` |
| link | `binary_record_with_same_protein` (relation between saturation-concentration rows and binary records of the same protein) |

A value that is not reported is empty and flagged in the `st_*` column; it is not imputed.

## Not included

Protein and nucleic-acid sequence text, language-model embeddings and free-text fields of the sources. Sequences are identified by UniProt accession and length.
The source resources have different licences (LLPSDB v2 states none; LLPSense lists GPL-3.0 and non-commercial no-derivatives terms; the PhaSepDB 3 pilot is an extraction from papers with their own licences), so check them before reusing the underlying records.
The tables are a processed view for benchmarking and are not a replacement for the source databases.

Version: 2026-11-03. Built from the records of the [records space](../records_space/README.md).
