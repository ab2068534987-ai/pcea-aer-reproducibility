# PCEA-AER Scheduler

This repository contains a curated code and data package for evaluating
PCEA-AER, a scheduling method for heterogeneous CPU-GPU DAG workflows under
time-varying power-envelope constraints.

The package includes source code, experiment configurations, processed
benchmark workflows, a trained checkpoint, evaluation scripts, and compact
result tables. It is not a full development workspace and does not include raw
Alibaba trace files or intermediate exploratory files.

## Repository Contents

| Path | Contents |
|---|---|
| `src/pcos/` | Scheduler implementation, environment, baseline heuristics, PPO policy, and AER module |
| `configs/` | Configuration files for the benchmark environment and PPO variants |
| `datasets/alibaba_pcea/processed_benchmark/` | Processed workflow JSON files and manifest CSV files |
| `models/pcea_aer/best_energy_pcea_ppo.pt` | Trained checkpoint used by the evaluation scripts |
| `scripts/` | Dataset construction and evaluation scripts |
| `results/` | Precomputed CSV result tables |
| `docs/` | Short notes on method, data, experiments, and results |
| `tests/` | Minimal smoke test for the scheduling environment |

## Dataset Information

The processed benchmark is stored under:

```text
datasets/alibaba_pcea/processed_benchmark/
