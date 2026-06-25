# Experiment Notes

The repository provides executable scripts for the two primary evaluation paths.

## Single-Workflow Evaluation

```powershell
$env:PYTHONPATH="src"
python scripts/evaluate_pcea_aer.py
```

Default inputs:

- Environment config: `configs/pcea_medium_cluster15_benchmark.json`
- PCEA-PPO config: `configs/train_pcea_medium_cluster15_benchmark_pcea_anti_idle.json`
- Checkpoint: `models/pcea_aer/best_energy_pcea_ppo.pt`
- Split: `benchmark`
- Limit: `136`

## Group-Size-5 Evaluation

```powershell
$env:PYTHONPATH="src"
python scripts/evaluate_group5_benchmark.py
```

The group benchmark submits five independent workflows at time zero to a shared `cluster_15_realistic` resource pool. The reported group-level energy is measured in the shared scheduling environment, not by summing independent single-workflow runs.

## Result Tables

CSV result tables are stored under `results/`. The most important files are:

- `results/01_main_group5_workflow_anti_idle_default/summary.csv`
- `results/01_main_group5_workflow_anti_idle_default/metrics.csv`
- `results/02_load_sensitivity_group1_5_10_anti_idle_default/load_summary.csv`
- `results/03_ablation_aer_anti_idle_default/ablation_aer_summary.csv`