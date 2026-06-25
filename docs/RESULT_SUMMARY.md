# Result Summary

This repository reports two primary evaluation scenarios: the single-workflow benchmark and the group-size-5 workload benchmark.

## Single-Workflow Benchmark

| Method | energy_active_total | deadline_miss_rate |
|---|---:|---:|
| HEFT | 70037.57 | 0 |
| PCEA | 70008.89 | 0 |
| PCEA-AER | 68951.03 | 0 |

PCEA-AER reduces active-server energy by **1.55%** relative to HEFT.

## Group-Size-5 Benchmark

| Method | group_energy_active_total | active_server_utilization |
|---|---:|---:|
| HEFT | 484057.08 | 0.66 |
| PCEA | 411597.17 | 0.69 |
| PCEA-AER | 397300.31 | 0.78 |

PCEA-AER reduces active-server energy by **17.92%** relative to HEFT.

## Key Result Files

- `results/01_main_group5_workflow_anti_idle_default/summary.csv`
- `results/01_main_group5_workflow_anti_idle_default/metrics.csv`
- `results/02_load_sensitivity_group1_5_10_anti_idle_default/load_summary.csv`
- `results/03_ablation_aer_anti_idle_default/ablation_aer_summary.csv`