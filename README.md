# Latent-Information Error Neutrality Simulator

Implementation based on:
- `DesignDocument/error_neutrality_design.tex`
- `DesignDocument/appendix_kitaev_chain_error_neutrality.tex`

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

For test/dev tooling:
```bash
pip install -r requirements-dev.txt
```

2. Run experiment:
```bash
python run_experiment.py --config configs/base.yaml
```

3. Outputs are written under:
```text
results/exp_<timestamp>/
  config_resolved.yaml
  metrics_seedwise.csv
  metrics.csv
  instancewise.csv
  instancewise_summary.csv
```

For topo runs, `metrics_seedwise.csv` and `metrics.csv` include:
- `accuracy`
- `balanced_accuracy`
- `error_rate`

## Implemented Modules

- `src/tasks/`
  - `obs_stability.py`: Task 1 observable-family neutrality (`N_Q`)
  - `topo_classification.py`: ED-based Kitaev-chain phase classification with observable operators
- `src/latents/`
  - `phi_drift.py`: AR(1) phase drift + coherent Z rotation
  - `gamma_coupling.py`: dephasing strength channel
- `src/inference/`
  - `kalman_phi.py`
  - `mle_gamma.py`
  - `none_inference.py`
- `src/controllers/`
  - `frame_comp.py`
  - `latent_reweight.py`
  - `none_controller.py`
  - `random_controller.py`
  - `oracle_controller.py`
- `src/runner.py`: unified experiment runner with seed/noise sweeps, CI aggregation, and `delta_vs_none`.
  - also emits `oracle_best` rows (`max(none, oracle)` with metric direction awareness)
  - also emits instance-level diagnostics for topo runs

## Two-Body Phase-1 Foundation

The repo now also includes a separate `src/twobody/` package for the `WECANDOIT.md` redesign track.
This package currently covers the Phase-1 building blocks:

- typed configs and observable specs
- state preparation families (`zero`, `rotated_product`, `bell`, `random_low_depth`)
- two-body Hamiltonian construction (`xx_zz`, `xy`, `ising_x`)
- exact or Lie-Trotter evolution circuits
- observable definitions and measurement-basis circuits
- density-matrix execution through Qiskit Aer
- coherent/stochastic noise composition and shot-based execution
- a minimal Bell-probe latent identifiability sweep
- observable-level reconstruction with `none / compensated / oracle` comparison
- feature extraction and feature-survival summaries driven by a shared Bell probe
- a phase-sensitive decision layer over `bell` vs `bell_i`
- integrated actionability/regime-map summaries across identifiability, reconstruction, feature survival, and decision
- Phase-5 robustness and ablation runners over the shared decision pipeline
- a unified measurement pipeline for latent estimation, feature extraction, and compensation
- fingerprint, transition-boundary, and mismatch/generalization experiments
- 2D transition-surface maps and stronger composite mismatch scenarios

Minimal demo:
```bash
python scripts/demo_twobody_density.py --config configs/twobody/base.yaml
```

Identifiability sweep:
```bash
python scripts/run_twobody_identifiability.py --config configs/twobody/identifiability.yaml
```

Reconstruction sweep:
```bash
python scripts/run_twobody_reconstruction.py --config configs/twobody/reconstruction.yaml
```

Feature-survival sweep:
```bash
python scripts/run_twobody_feature_survival.py --config configs/twobody/feature_survival.yaml
```

Decision sweep:
```bash
python scripts/run_twobody_decision.py --config configs/twobody/decision.yaml
```

Actionability summary and regime map:
```bash
python scripts/summarize_twobody_actionability.py --results-dir results/twobody_decision_<timestamp>
python scripts/plot_twobody_regime_map.py --actionability-csv results/twobody_decision_<timestamp>/actionability_threshold_shot_8192.csv
```

Integrated regime-map summary:
```bash
python scripts/run_twobody_regime_map.py --identifiability-dir results/twobody_identifiability_<timestamp> --reconstruction-dir results/twobody_reconstruction_<timestamp> --feature-dir results/twobody_feature_survival_<timestamp> --decision-dir results/twobody_decision_<timestamp> --backend-type shot --shots 8192 --classifier threshold --feature-name phase_sin_component --label-left bell --label-right bell_i
python scripts/plot_twobody_regime_summary.py --regime-map-csv results/twobody_regime_map_<timestamp>/regime_map_summary.csv
```

Phase-5 robustness and ablation:
```bash
python scripts/run_twobody_robustness.py --config configs/twobody/robustness.yaml
python scripts/plot_twobody_robustness.py --results-dir results/twobody_robustness_<timestamp> --scenario measurement_noise --classifier threshold --shots 2048
python scripts/run_twobody_ablation.py --config configs/twobody/ablation.yaml
python scripts/plot_twobody_ablation.py --results-dir results/twobody_ablation_<timestamp> --group feature_set --classifier threshold --shots 8192
```

Fingerprint, transition, and mismatch:
```bash
python scripts/run_twobody_fingerprint.py --config configs/twobody/fingerprint.yaml
python scripts/plot_twobody_fingerprint.py --results-dir results/twobody_fingerprint_<timestamp> --backend-type shot --shots 8192
python scripts/run_twobody_transition.py --config configs/twobody/transition.yaml
python scripts/plot_twobody_transition.py --results-dir results/twobody_transition_<timestamp> --backend-type shot --shots 8192
python scripts/run_twobody_mismatch.py --config configs/twobody/mismatch.yaml
python scripts/plot_twobody_mismatch.py --results-dir results/twobody_mismatch_<timestamp> --classifier threshold --shots 2048
```

2D transition surfaces and composite mismatch:
```bash
python scripts/run_twobody_transition_surface.py --config configs/twobody/transition_surface.yaml
python scripts/plot_twobody_transition_surface.py --results-dir results/twobody_transition_surface_<timestamp> --map-name gamma_vs_amplitude --backend-type shot --shots 2048
python scripts/plot_twobody_transition_surface.py --results-dir results/twobody_transition_surface_<timestamp> --map-name gamma_vs_measurement --backend-type shot --shots 2048
python scripts/run_twobody_mismatch.py --config configs/twobody/mismatch_composite.yaml
python scripts/plot_twobody_mismatch.py --results-dir results/twobody_mismatch_<timestamp> --classifier threshold --shots 2048
```

Paper-style decomposition / residual / limit figures:
```bash
python scripts/run_twobody_paper_figures.py --config configs/twobody/paper_figures.yaml
python scripts/plot_twobody_paper_figures.py --results-dir results/twobody_paper_figures_<timestamp> --backend-type shot --shots 4096
```
`run_twobody_paper_figures.py` also computes a density-based full upper bound oracle in addition to the structured phi/gamma oracle, so the paper plots can separate latent-model recovery from the absolute noiseless limit.
The default paper config now uses a harder decision task: `bell_i` vs `rotated_product` under target evolution (`evolution_time=0.4`), while keeping the Bell probe on a calibration-valid `evolution_time=0.0` path.

Plotting examples:
```bash
python scripts/plot_twobody_identifiability.py --results-dir results/twobody_identifiability_<timestamp>
python scripts/plot_twobody_reconstruction.py --results-dir results/twobody_reconstruction_<timestamp>
python scripts/plot_twobody_feature_survival.py --results-dir results/twobody_feature_survival_<timestamp>
```

## Tests

- Deterministic oracle recovery: `tests/test_phi_oracle_deterministic.py`
- Monte Carlo oracle improvement: `tests/test_phi_oracle_mc.py`

Run (if `pytest` is installed):
```bash
python -m pytest -q tests
```

## Feedback Runs

- Short topo + gamma validation run:
```bash
python run_experiment.py --config configs/experiments/topo_gamma_feedback.yaml
```

- Gamma=0.5 diagnostic run with `instancewise.csv`:
```bash
python run_experiment.py --config configs/experiments/topo_gamma_diagnostic_gamma05.yaml
```

- F3-only threshold sweep:
```bash
python scripts/sweep_topo_threshold.py --config configs/experiments/topo_gamma_diagnostic_gamma05_f3.yaml
```

- Offline `f2+f3` grid-search from existing `instancewise.csv`:
```bash
python scripts/grid_search_topo_f23.py --instancewise results/exp_<timestamp>/instancewise.csv
```

This analysis writes:
```text
results/analyses/f23_grid_<timestamp>/
  best_params.csv
  coarse_topk.csv
  test_eval.csv
  baseline_eval.csv
  test_eval_seedwise.csv
  test_eval_summary.csv
  train_feature_stats.csv
  gamma_curve.csv
  gamma_curve_seedwise.csv
  gamma_curve_summary.csv
```
