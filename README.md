
`main_text/davies_benchmark` contains the Davies many-body benchmark behind Fig. 2 in the main text. Running `python generate_davies_many_body_figures.py` creates `davies_many_body_bundle_panels.pdf` and `davies_many_body_mixing_concentration.pdf` inside the local `outputs` folder.

`supplement/S16_S17_logical_sector` contains the local-qubit logical-sector model used in the supplement. Running `python generate_logical_sector_figures.py` creates `appendixM_logical_sector_schematic.pdf`, `appendixM_logical_sector_overlap.pdf`, `appendixM_logical_sector_bundle.pdf`, and `appendixM_logical_sector_scaling.pdf`. 

`supplement/S18_skin_fixed_rate` contains the fixed-rate single-particle skin baseline used for the threshold-comparison figures in Sec. S18. Running `python generate_skin_threshold_comparison.py` creates `figure_open_gap_strong_skin_six_bundle_threshold_comparison.pdf` and `figure_open_gap_strong_skin_gap_scaling_threshold_comparison.pdf`. 

`supplement/S18_boundary_fixed_rate` contains the fixed-rate many-body boundary-supported baseline from the same supplement section. Running `python generate_boundary_fixed_rate_figures.py` creates `open_gap_constant_bundle_3x2.pdf` and `open_gap_constant_linear_separation.pdf`.


