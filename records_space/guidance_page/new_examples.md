# Guidance tab: new examples (generated 2026-11-13)

All numbers below are read from the json files in this folder; nothing is typed by hand. Sampling ran on WSL (CPU only, `nice -n 10`, 4 threads, pixi `~/t3ebm`: torch 2.14.1 CPU, torchebm 0.9.1) with the same settings as the existing examples (`scripts/rs70_yc_trajectories.py`, `scripts/rs95_arc_hmc_runs.py`); figures and GIFs were drawn locally with `~/pixi/figs` (`scripts/rs131b_yc_new_render.py`, `scripts/rs132b_iso_gifs.py`, copies of rs73 / rs96) and compressed with ffmpeg (`palettegen` 64 colours, no dither, frame count and per-frame delays unchanged).

## (y, c) sets (`yc/`)

Settings as rs70: σ = 0.16 TorchEBM conditional energy, Langevin η = 0.25σ², T = 0.5, 1,200 steps, w = 4, K = 16 chains per start record (at most 400 start records), controls = no guidance and guidance to the same label; GIF = 24 chains, one frame per 10 steps, 121 frames, 60-step trail, black final dots held about 8 s.

| id | records y=1 / y=0 | sequences | proteins | papers | map distance of the two label groups (SD) | median Δr, 0→1 guided (none) | median Δr, 1→0 guided (none) | chains ending ≤ 0.10 SD of a target record, 0→1 / 1→0 guided (none) |
|---|---|---|---|---|---|---|---|---|
| `tau_peg` | 21 / 25 | 9 | 1 | 1 | 0.03 | +0.68 (+0.09) | -0.95 (+0.07) | 1% / 2% (2% / 2%) |
| `fus` | 28 / 20 | 26 | 22 | 1 | 0.02 | +0.55 (-0.06) | -0.83 (-0.04) | 0% / 0% (1% / 0%) |
| `ubqln2` | 7 / 9 | 9 | 1 | 1 | 0.08 | +0.01 (-0.52) | -1.38 (-0.57) | 1% / 0% (1% / 0%) |
| `zo1` | 7 / 10 | 12 | 3 | 1 | 0.01 | +0.90 (+0.37) | -0.58 (+0.37) | 0% / 0% (1% / 0%) |
| `g3bp1` | 6 / 5 | 4 | 1 | 1 | 0.01 | +0.58 (-0.02) | -0.47 (-0.11) | 0% / 0% (0% / 0%) |
| `hnrnpa2b1` | 5 / 7 | 8 | 4 | 1 | 0.11 | +0.37 (+0.05) | -1.06 (-0.19) | 2% / 1% (0% / 5%) |

Things to keep in mind (all six):
- Every set is **one paper only** (`n_paper` = 1): the cell is a within-paper cell, so label differences inside it are differences between that paper's constructs or conditions.
- The two label groups sit almost on top of each other on the map (distance 0.01–0.11 SD, mean-r difference ≈ 0). Guidance moves chains in r in the intended direction (compare with the no-guidance column), but chains almost never end on a target-label record (≤ 2%), the same picture as the existing hnRNPDL and BRD4 examples: the 2-D field carries no condition c.
- Small sets: `g3bp1` (6 / 5 records, 4 sequences), `hnrnpa2b1` (5 / 7), `ubqln2` (7 / 9), `zo1` (7 / 10). `fus` is a FUS-family variant/chimera set (22 UniProt entries), `zo1` has 3 proteins, `hnrnpa2b1` 4.
- `ubqln2` 0→1: guided Δr is about 0 because the no-guidance chains drift to lower r; guidance only cancels that drift.

## Iso-lines (`iso/`)

Rule (from `scripts/rs80_isocontours.py`, with one added guard): pass = rs80 score > 0 (purity < 1, arc >= 1.0 SD, >= 10 records of each label within 0.12 SD of the line) AND r changes sign along the line AND >= 5 papers near the line; new lines ranked by purity (lower = more homogeneous), at most 8.
Re-evaluating the 23 rs80 grid rows reproduces `rs80_table.tsv` (max abs difference: arc 1.1e-07, purity 0.0e+00, n_near 0, score 1.4e-08).

Settings as rs95: line parametrised by arc length, 1-D HMC (ε = 0.05, 10 leapfrog steps, T = 0.5, w = 4), 400 proposals, 60 equal-arc starts × 8 = 480 chains per direction; GIF = 4 proposals per frame, 5 fading ghosts, black final dots held about 8 s.

| id | line | from | purity (lower = more homogeneous) | length (SD) | y=1 / y=0 records near the line | papers | segments | TV to y=1 / y=0 | accept rate to y=1 / y=0 | median arc move (SD) to y=1 / y=0 | off-support chains to y=1 / y=0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `conc32` | Iso-concentration 31.6 µM | rs80 grid | 0.73 | 18.0 | 1312 / 866 | 130 | 7 | 0.15 / 0.13 | 0.81 / 0.79 | 0.68 / 0.79 | 3.3% / 3.3% |
| `len200` | Iso-length 200 aa | requested list | 0.74 | 15.3 | 1249 / 556 | 108 | 6 | 0.14 / 0.20 | 0.77 / 0.76 | 0.82 / 0.60 | 3.1% / 11.2% |
| `conc3p2` | Iso-concentration 3.16 µM | rs80 grid | 0.77 | 8.5 | 766 / 413 | 81 | 7 | 0.08 / 0.08 | 0.89 / 0.66 | 0.34 / 0.38 | 0.8% / 2.5% |
| `nacl50` | Iso-NaCl 50 mM | requested list | 0.83 | 21.9 | 2277 / 1369 | 228 | 5 | 0.16 / 0.23 | 0.94 / 0.89 | 1.31 / 1.14 | 1.5% / 4.0% |
| `conc1` | Iso-concentration 1 µM | requested list | 0.92 | 2.8 | 185 / 117 | 8 | 4 | 0.07 / 0.08 | 0.58 / 0.55 | 0.21 / 0.21 | 3.8% / 9.4% |

Candidates that were evaluated but not sampled (full numbers in `iso/_candidates.json`):

| line | requested | status | why |
|---|---|---|---|
| Iso-length 50 aa | no (rs80 grid) | fails | not more homogeneous than the whole data (purity 1.11 ≥ 1) |
| Iso-length 100 aa | no (rs80 grid) | fails | not more homogeneous than the whole data (purity 1.07 ≥ 1) |
| Iso-length 800 aa | yes | fails | no segment inside the data-supported area |
| Iso-NaCl 0 mM | no (rs80 grid) | fails | no segment inside the data-supported area |
| Iso-NaCl 300 mM | yes | fails | line too short (0.83 SD < 1.0); too few records near the line (y=1 44, y=0 0; need ≥ 10 each); only 1 papers near the line (need ≥ 5) |
| Iso-NaCl 500 mM | no (rs80 grid) | fails | line too short (0.67 SD < 1.0); only 1 papers near the line (need ≥ 5) |
| Iso-temperature 4 °C | yes | fails | only 3 papers near the line (need ≥ 5) |
| Iso-temperature 37 °C | yes | fails | not more homogeneous than the whole data (purity 1.47 ≥ 1) |
| Iso-temperature 60 °C | no (rs80 grid) | fails | no segment inside the data-supported area |
| Iso-pH pH 5 | no (rs80 grid) | fails | no segment inside the data-supported area |
| Iso-pH pH 6 | yes | fails | line too short (0.92 SD < 1.0); only 1 papers near the line (need ≥ 5) |
| Iso-pH pH 6.5 | no (rs80 grid) | fails | not more homogeneous than the whole data (purity 1.25 ≥ 1) |
| Iso-pH pH 8 | yes | fails | not more homogeneous than the whole data (purity 1.20 ≥ 1) |
| Iso-pH pH 8.5 | no (rs80 grid) | fails | not more homogeneous than the whole data (purity 1.29 ≥ 1) |
| Iso-concentration 100 µM | yes | fails | not more homogeneous than the whole data (purity 1.04 ≥ 1) |
| Iso-concentration 316 µM | no (rs80 grid) | fails | not more homogeneous than the whole data (purity 1.11 ≥ 1) |
| Iso-crowder 2 % | no (rs80 grid) | fails | not more homogeneous than the whole data (purity 1.32 ≥ 1) |
| Iso-crowder 5 % | no (rs80 grid) | fails | not more homogeneous than the whole data (purity 1.10 ≥ 1) |
| Iso-crowder 10 % | no (rs80 grid) | fails | line too short (0.54 SD < 1.0); only 2 papers near the line (need ≥ 5) |

Existing lines (already have GIFs, re-evaluated only as a check): Iso-length 400 aa (purity 0.66), Iso-NaCl 150 mM (purity 0.49), Iso-temperature 22 °C (purity 0.55), Iso-pH pH 7.5 (purity 0.68), Iso-concentration 10 µM (purity 0.68), Iso-crowder 0.5 % (purity 0.98).
