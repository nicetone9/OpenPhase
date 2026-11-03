<p align="center">
  <a href="https://nicetone9.github.io/OpenPhase/"><img src="assets/openphase-main.webp" alt="OpenPhase: from a system and its conditions to prediction, phase diagram and generation" width="100%"></a>
</p>

<p align="center">
  <b><a href="https://nicetone9.github.io/OpenPhase/">Open the OpenPhase site</a></b> &nbsp;·&nbsp;
  <a href="https://nicetone9.github.io/OpenPhase/records_space/">Records and protein universe (P)</a> &nbsp;·&nbsp;
  <a href="https://nicetone9.github.io/OpenPhase/records_space/negatives/">Negatives and unlabelled (N + U)</a>
</p>

# OpenPhase

OpenPhase: Condition-Aware Exploration of Biomolecular Phase Separation
OpenPhase is an open-source platform and dataset designed to accelerate knowledge discovery in Liquid-Liquid Phase Separation (LLPS). It is the first condition-aware platform for system-level exploration of how environmental factors influence biomolecular phase behavior.   

## 🌐 Interactive site

**[nicetone9.github.io/OpenPhase](https://nicetone9.github.io/OpenPhase/)** is the home page of the OpenPhase data resources. Click a part of the system figures to open the explorer on that system; every page has a light / dark switch and runs entirely in the browser.

| System | What it holds | Page |
|---|---|---|
| **P** | 11,439 single-protein experiment records in one map, plus 6,338 proteins that the tools list as positives but that have no record (hollow rings) | [records_space/](https://nicetone9.github.io/OpenPhase/records_space/) |
| **N + U** | 35,107 proteins: the training negatives of ten tools and PPMC against a Swiss-Prot reference sample. No positives, no conditions | [records_space/negatives/](https://nicetone9.github.io/OpenPhase/records_space/negatives/) |

Multi-protein systems (protein(2), protein(3), complexes) are shown on the home page but are not in the explorer yet. Details, data notes and licences: [`records_space/README.md`](records_space/README.md).

## 🚀 Key Features

### Condition-Aware Modeling
Moves beyond simple sequence prediction to incorporate extrinsic environmental factors (salt concentration, temperature, pH, etc.) that critically influence phase separation behavior.

### Comprehensive Datasets, organised by system

The records are organised by **system** (the set of components of an experiment, without concentrations); see [`data/`](data/README.md).

| System | Records | Folder |
|---|---:|---|
| protein only | 7,490 | `data/protein/` |
| protein + RNA | 2,511 | `data/protein_RNA/` |
| protein + DNA | 662 | `data/protein_DNA/` |
| protein, saturation concentration | 776 | `data/protein_csat/` |

Records come from LLPSDB v2, LLPSense, RNAPSEC, a PhaSepDB 3 pilot extraction, Bremer 2022 and Ausserwöger 2025, merged and de-duplicated. A nucleic acid is one of the conditions of a record. The three original sources keep their loaders:

- **db1** (`db1.py`): processed LLPSDBv2 data containing proteins, RNA, and DNA sequences with phase separation annotations
- **db2** (`db2.py`): manually collected records from laboratory notebooks covering both in vitro and in vivo experiments (not in the tables of `data/`)
- **db3** (`db3.py`): preprocessed RNAPSEC data focused on protein-RNA interaction systems and their phase behavior

### Standardized Benchmarking
Formalized tasks and evaluation protocols for:
- Outcome prediction
- Condition inference
- Protein design
- Reproducible research in LLPS

## 🛠️ Canonical Tasks

### 1. Phase Outcome Prediction
Predicting whether a system undergoes LLPS (y ∈ {0, 1}) given:
- Components (x): protein/RNA sequences
- Conditions (c): environmental factors (pH, salt, temperature, etc.)

### 2. Condition Inference
Inferring the potential experimental conditions (c) that lead to a specific phase separation outcome, enabling rational experimental design.

### 3. Phase System Design
Generating potential protein sequences (x) that exhibit desired phase behaviors under target conditions using generative models.

## 📂 Repository Structure

```
OpenPhase/
├── data/                  # records by system (protein, protein_RNA, protein_DNA, protein_csat)
├── openphase_systems.py   # load the records by system
├── db1.py                 # LLPSDBv2 dataset interface (one source)
├── db2.py                 # Laboratory notebook dataset (one source)
├── db3.py                 # RNAPSEC dataset interface (one source)
├── Discrete/              # Discrete diffusion model
├── ldm/                   # Latent diffusion model components
├── MTDP/                  # MTDP embedder
├── records_space/         # records space and protein universe explorer
├── meta_config.py         # Configuration utilities
├── meta_embed.py          # Embedding generation
└── params_all.yaml        # Global hyperparameters
```

## 🔧 Installation

```bash
git clone https://github.com/nicetone9/OpenPhase.git
cd OpenPhase
# Install dependencies (see requirements.txt or setup.py)
```

## 📊 Getting Started

Load the records by system:

```python
from openphase_systems import list_systems, load_system

list_systems()                         # records, sequences, papers and sources of each system
rna = load_system("protein_RNA")       # a pandas data frame; load_system("protein_RNA", source="RNAPSEC")
```

Build the embedded datasets of the three original sources:

```python
from db1 import DB1Dataset
from db2 import DB2Dataset
from db3 import DB3Dataset

db1 = DB1Dataset()
db2 = DB2Dataset()
db3 = DB3Dataset()
```

## 📚 Tasks & Benchmarks

Each canonical task includes:
- Standardized train/test splits
- Baseline models
- Evaluation metrics
- Documentation in `task1.py`, `task2.py`, `task3.py`

## 🤝 Contributing

Contributions are welcome! Please refer to contribution guidelines and open issues for ongoing research directions.

## 📖 Citation

If you use ConditionPhase in your research, please cite the associated publication.

## 📧 Contact

For questions, suggestions, or collaborations, visit the [GitHub repository](https://github.com/nicetone9/ConditionPhase) and contact [email](peiran@cmu.edu).
