# simu_code

Numerical reproducibility package for the current draft of **“Typical Mixing and Rare-State Bottlenecks in Open Quantum Systems.”**

This folder contains only the scripts needed to regenerate the numerical figures used in the paper draft. Cached outputs, generated figure PDFs, copied figure files, exploratory notebooks, older `.npz` working data, and slide-style schematic sources from `ppt` are intentionally not included.

## Scope of this package

The current draft separates three numerical roles:

1. A finite-size Davies benchmark for vertical fixed-time concentration and horizontal threshold-time concentration.
2. A logical/protected-sector example where Haar-typical states mix on the fast bulk scale while rare prepared states follow a slow leakage scale.
3. Fixed-rate skin and boundary-supported baselines used to isolate overlap geometry from additional Liouvillian-gap closing.

The scripts reproduce the numerical illustrations. The concentration theorems, the one-mode bottleneck estimate, and the beyond-Haar transfer bounds are proved in the paper and Supplemental Material, not by the code.

## Layout and figure map

### `main_text/davies_benchmark`

Davies many-body benchmark for **Fig. 2** of the main text.

Run:

```bash
cd main_text/davies_benchmark
python generate_davies_many_body_figures.py
```

The script writes the following files to the local `outputs` directory:

- `davies_many_body_bundle_panels.pdf`
- `davies_many_body_mixing_concentration.pdf`

This example is a product Davies model with a unique product Gibbs state. It is used only to illustrate the concentration-to-crossing mechanism; it is not a rare-state bottleneck example.

### `supplement/S16_S17_logical_sector`

Local-qubit logical-sector model used in the current Supplemental Material, especially the logical-sector and protected-sector discussion in **Secs. S7.4–S7.5** and **Figs. S1–S4**. The folder name is historical and refers to an older appendix numbering.

Run:

```bash
cd supplement/S16_S17_logical_sector
python generate_logical_sector_figures.py
```

The script writes the following files to the local `outputs` directory:

- `appendixM_logical_sector_schematic.pdf`
- `appendixM_logical_sector_overlap.pdf`
- `appendixM_logical_sector_bundle.pdf`
- `appendixM_logical_sector_scaling.pdf`

The model contains one weakly leaking logical qubit and a fast syndrome register. The numerical purpose is to show that the high-probability mixing scale can remain fast while a prepared logical state produces a much slower worst-case tail.

### `supplement/S18_skin_fixed_rate`

Fixed-rate single-particle skin baseline used in the current Supplemental Material, **Sec. S8.1** and **Figs. S5–S6**. The folder name is historical and refers to an older appendix numbering.

Run:

```bash
cd supplement/S18_skin_fixed_rate
python generate_skin_threshold_comparison.py
```

The script writes the following files to the local `outputs` directory:

- `figure_open_gap_strong_skin_six_bundle_threshold_comparison.pdf`
- `figure_open_gap_strong_skin_gap_scaling_threshold_comparison.pdf`

The helper modules in this folder generate the dynamics directly, so this section no longer depends on cached `.npz` data from older working folders.

This baseline is kept at fixed rate to separate the overlap mechanism from gap-closing effects. The threshold comparison is intentional: the lower threshold is closer to the clean one-mode tail, while the larger threshold still contains visible subleading-mode corrections.

### `supplement/S18_boundary_fixed_rate`

Fixed-rate many-body boundary-supported baseline used in the current Supplemental Material, **Sec. S8.2** and **Figs. S7–S8**. The folder name is historical and refers to an older appendix numbering.

Run:

```bash
cd supplement/S18_boundary_fixed_rate
python generate_boundary_fixed_rate_figures.py
```

The script writes the following files to the local `outputs` directory:

- `open_gap_constant_bundle_3x2.pdf`
- `open_gap_constant_linear_separation.pdf`
- small summary tables in the local `outputs` directory

This example keeps the boundary rate fixed, with the displayed branch having a nonclosing gap. The growing separation is therefore tied to the shrinking typical boundary overlap rather than to a vanishing relaxation rate. The reported worst-case numerical benchmark is obtained by scanning the computational-basis product states in the diagonal sector, as described in the Supplemental Material.

## Dependencies

The scripts were written for Python 3.11.

Required packages:

```bash
pip install numpy matplotlib scipy pandas
```

Folder-specific usage:

- `main_text/davies_benchmark`: `numpy`, `matplotlib`
- `supplement/S16_S17_logical_sector`: `numpy`, `matplotlib`
- `supplement/S18_skin_fixed_rate`: `numpy`, `matplotlib`, `scipy`
- `supplement/S18_boundary_fixed_rate`: `numpy`, `matplotlib`, `scipy`, `pandas`

## Reproducibility notes

No external data files are needed. Each script builds its figures from the model definition and the sampling procedure in that folder.

All generated files are written to a local `outputs` directory. To keep the package tree clean after a run, remove those `outputs` directories before committing unless the release explicitly requires generated figures.

The fixed-rate skin and boundary folders should be read together with the Supplemental Material discussion of the local scope of the crossing theorem. The numerical comparisons there are threshold-dependent checks of the overlap mechanism; they are not claims that every threshold is already in the final asymptotic one-mode regime.
