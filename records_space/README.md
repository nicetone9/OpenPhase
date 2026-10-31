# Records space

An interactive map of 11,439 single-protein LLPS experiment records (10,663 binary records and 776 csat records).
Each dot is one experiment: a protein, an optional nucleic acid, and the conditions it was tested under.

**Open the explorer:** <https://nicetone9.github.io/OpenPhase/records_space/>
(or download [`index.html`](index.html) and open it in a browser; it is one self-contained file and runs entirely in the page, nothing is sent anywhere. Only the fonts are loaded from Google Fonts.)

## How the map is built

Distance between two records is a weighted sum of four blocks:

| block | weight | content |
|---|---|---|
| protein | 0.40 | ESMC-600M layer-18 embedding of the sequence (z-score, PCA to 64 dimensions, cosine) |
| nucleic acid | 0.15 | none vs present, class (RNA/DNA), name classes, length, concentration, ratio, sequence and fold features |
| conditions | 0.35 | protein concentration, pH, temperature, salts, crowders |
| buffer and additives | 0.10 | buffer family and additive families |

A value that is not reported counts as its own state; it is not imputed. Labels, csat values and source are not inputs; they only colour the dots.
The map is a t-SNE of this distance (UMAP views are alternatives). The weights and the missing-value penalty were not tuned.

## What the explorer does

* **Explore:** colour by protein, conditions, readout or source; filters; three map styles (solid blobs, soft glow, dots only); outline styles (light edge, coloured edge, dashed edge, contour rings, whole-map outline, soft halo).
* **Highlight tool entries:** the entries of LLPSense, Droppler (the LLPSDB experiments in its supplementary Table S2) and RNAPSEC, plus LLPSDB v2, PhaSepDB 3 (LLM pilot), Bremer 2022 and Ausserwöger 2025, are highlighted with saturated dots.
  Tools that only take a protein (the PPMC benchmark set and the training sets of catGRANULE 2.0, FuzDrop, LLPhyScore, LLPS-predict, OptPredLLPS, PPT-taskA, ProtGPS, PSAP, PScalpel, PSPHunter and PSPire) highlight the records of the same proteins with light dots and are marked "no conditions". Their proteins that are not in the map are drawn in the protein universe view (next bullet).
* **Protein universe:** the *Map* menu has a *Protein universe (UMAP)* view. The 6,338 proteins that the tools mention but that have no record in the map (the PPMC drivers and clients, the LTP and HTP benchmark sets and the training positives of the 11 predictors above) are drawn as **hollow rings** in the same ESMC protein space; records, which have conditions, are **filled dots**. Highlighting a protein-only tool switches to this view. A ring gives the protein, its length and organism, the tools that list it and the closest protein with records, and one click loads it into *Try your own experiment* (its stored embedding is used; the neighbours come from the protein alone).
  Sequences were fetched from UniProt (156 entries that no longer exist in UniProtKB were taken from UniParc; 5 could not be found), embedded with ESMC-600M and projected into the dataset's protein space with the dataset's own z-score and PCA, then laid out with UMAP together with the dataset sequences. Negatives and unlabelled proteins are not in this view.
* **Try your own experiment:** paste a sequence, describe the conditions in words or fill the boxes, or load an entry from a tool. The page parses the conditions with the rules of the dataset's own parser, computes the distance to all records and shows the nearest records, how well that region is covered and the label share among the 15 nearest.
  It does **not** predict phase separation: neighbours from the same paper share the label (AUROC 0.84), neighbours from other papers and proteins barely do (0.50 with protein similarity, 0.59 with conditions only).
  New sequences get an exact ESMC embedding only when they are in the dataset or are one of the built-in examples; any other pasted sequence gets a rough in-browser approximation (composition and dipeptide regression), and the page says so.

## Negatives and unlabelled proteins (`negatives/`)

A second page, [`negatives/index.html`](negatives/index.html), shows only negatives and unlabelled proteins (no positives) in one ESMC-600M space, with no conditions anywhere:
35,107 proteins, 23,433 of them in at least one negative set (the training negatives of ten predictors and PPMC NP / ND) and 11,674 unlabelled proteins from a Swiss-Prot reference sample (12,000 random + 4,000 human proteins).
The page is grey and light grey-blue throughout. Pick a negative set to highlight it, colour the negatives by the number of sets that list them (17,194 of them are in two or more), show densities, search by UniProt, gene or name, and click a protein for the sets it belongs to. A ring marks proteins that also have records with conditions in the main explorer.
Proteins were embedded as described above (UniProt, UniParc for entries that no longer exist) and laid out with UMAP using only these proteins; UMAP keeps neighbourhoods, not distances. The Swiss-Prot sample is a background, not a set of verified negatives, and training negatives are mostly proteins that were not listed as phase-separating. The set memberships come from the published training sets as collected in a benchmarking project; check each source's licence before reusing them.

## Site and links

The pages are served by GitHub Pages from this repository: `index.html` (home, with the two system figures), `records_space/` (the explorer) and `records_space/negatives/` (negatives and unlabelled proteins). The home page is built by `scripts/rs17_build_site.py` from `home_page/home_template.html`. Each page has a light / dark switch; the choice is remembered in the browser and shared by the three pages.

The explorer takes its starting state from the address, so a link can open it on a system or a source:

| parameter | values | example |
|---|---|---|
| `na` | `none`, `RNA`, `DNA` | `#na=RNA` |
| `csat` | `only` (csat records), `hide` (binary records only) | `#na=RNA&csat=hide` |
| `hl` | a tool or database: `llpsense`, `droppler`, `rnapsec`, `llpsdb`, `phasepdb3`, `bremer`, `ausserwoger`, or a protein-only set | `#hl=rnapsec` |
| `protein` | a protein name from the colour legend | `#protein=FUS` |
| `q` | text for the find box | `#q=P35637` |
| `lens` | `tsne_full`, `umap_full`, `umap_prot`, `umap_cond`, `umap_fullM`, `umap_universe` | `#lens=umap_universe` |

## Folder

```
index.html            the explorer (built by scripts/rs05_build_explorer.py); the site home page is ../index.html
explorer/             page template, the three JavaScript modules (condition parser, space, panel), tool entry table, protein universe table (embeddings, names, tool masks), example definitions
export/               arrays the page embeds: record conditions, k-mer sketches and ESMC-derived embeddings of the dataset sequences, parameters
scripts/              input building, page building, example and tool tables (paths are those of the original machine)
colab/                code that ran on Colab (embeddings, distances, t-SNE/UMAP, figures, export)
negatives/            the negatives and unlabelled proteins page (built by scripts/rs16_build_pu.py)
figs/                 static figures of the map
```

## Data notes

The records come from LLPSDB v2, LLPSense, RNAPSEC, PhaSepDB 3 (LLM pilot), Bremer 2022 and Ausserwöger 2025.
The page holds, for each record, the protein name, UniProt number, paper or entry identifier, label and conditions, plus, for the dataset sequences, ESMC-derived embeddings and sampled 5-mer hash sketches (a fingerprint used to recognise a pasted sequence, not the sequence itself). It holds no protein sequence text except six UniProt examples.
Protein names and organisms of the universe proteins come from UniProt (CC BY 4.0). ESMC-600M has its own licence terms and the page holds embeddings derived from it.
The source resources have different licences (LLPSDB v2 states none; LLPSense lists GPL-3.0 and non-commercial no-derivatives terms), so check them before reusing the underlying records.
The full per-record tables (`out/rs_plot_table.tsv`, `out/rs_coords.tsv`) that `scripts/rs05_build_explorer.py` reads are not included, so the page cannot be rebuilt from this folder alone.

Version: 2026-10-30.
