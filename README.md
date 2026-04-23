

`main_text/davies_benchmark` contains the Davies many-body benchmark behind Fig. 2 in the main text. Running `python generate_davies_many_body_figures.py` creates `davies_many_body_bundle_panels.pdf` and `davies_many_body_mixing_concentration.pdf` inside the local `outputs` folder.

`supplement/S16_S17_logical_sector` contains the local-qubit logical-sector model used in the supplement. Running `python generate_logical_sector_figures.py` creates `appendixM_logical_sector_schematic.pdf`, `appendixM_logical_sector_overlap.pdf`, `appendixM_logical_sector_bundle.pdf`, and `appendixM_logical_sector_scaling.pdf`. These files match the figures used in the logical-sector discussion and in the numerical settings section.

`supplement/S18_skin_fixed_rate` contains the fixed-rate single-particle skin baseline used for the threshold-comparison figures in Sec. S18. Running `python generate_skin_threshold_comparison.py` creates `figure_open_gap_strong_skin_six_bundle_threshold_comparison.pdf` and `figure_open_gap_strong_skin_gap_scaling_threshold_comparison.pdf`. The helper modules in the same folder generate the dynamics directly, so this section no longer depends on cached `.npz` data from the older working folders.

`supplement/S18_boundary_fixed_rate` contains the fixed-rate many-body boundary-supported baseline from the same supplement section. Running `python generate_boundary_fixed_rate_figures.py` creates `open_gap_constant_bundle_3x2.pdf` and `open_gap_constant_linear_separation.pdf`, together with small summary tables in the local `outputs` folder.

## Dependencies

The scripts were written for Python 3.11. The Davies and logical-sector folders use `numpy` and `matplotlib`. The fixed-rate skin and boundary folders use `numpy`, `matplotlib`, and `scipy`. The boundary folder also uses `pandas` for the summary tables written during figure generation.

## Notes

No external data files are needed. Each script builds its figures from the model definition and the random sampling carried out inside that folder. If you want the package tree to stay clean after a run, you can remove the `outputs` directories before committing.
