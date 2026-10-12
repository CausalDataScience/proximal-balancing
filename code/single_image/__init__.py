"""Single-image PROBE, phase 1 (plan memo/2026-10-08-single-image-probe-plan-v1.md, sections 6, 8.1, 10.1-10.2).

Modules: regions (fixed lobe masks and adjacency), phantom (generator), summaries (fixed chi_l), reader
(region readers psi_k and their diagnostics), mae (masked autoencoder), nets (shared CNN trunk), folds,
oracle (population quantities), level_a (PROBE on reader blocks and comparators), run_phase1 (CLI).
"""
