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
```

The grouped benchmark used by the main evaluation contains 135 processed
workflows, arranged as 27 groups of 5 workflows.

Each workflow is stored as a JSON file. Manifest CSV files list workflow paths
and split information with the following columns:

```text
workflow_path, format, split, repeat, weight, workflow_id
```

Each workflow JSON contains:

- `workflow_id`: workflow identifier.
- `submit_time`: workflow submit time.
- `tasks`: DAG task records.
- `edges`: precedence edges between tasks.
- `makespan_target`: target completion time used by the scheduler.
- `metadata`: derived DAG properties.

Task records include fields such as:

```text
id, cpu, gpu, mem, base_duration, gpu_intensity, data_size,
output_mb, profile_name, preferred_type, deadline
```

Edge records include:

```text
src, dst, data_mb
```

The processed workflows are derived from Alibaba Cluster Trace 2018. Raw Alibaba
CSV files are not included in this repository.

## Code Information

Important implementation files:

- `src/pcos/env/scheduler_env.py`: scheduling environment and episode metrics.
- `src/pcos/env/power_envelope.py`: time-varying power-envelope model.
- `src/pcos/baselines/heuristics.py`: baseline scheduling policies.
- `src/pcos/rl/pcea_ppo.py`: PPO-based scheduling policy.
- `src/pcos/postprocess/active_window_energy_reallocation.py`: AER post-processing module.
- `scripts/evaluate_group5_benchmark.py`: grouped workload evaluation.
- `scripts/evaluate_pcea_aer.py`: single-workflow evaluation.
- `scripts/build_alibaba_pcea_dataset.py`: optional dataset construction script.

## Requirements

- Python 3.10 or newer
- `numpy>=1.22`
- `pandas>=1.5`
- `torch>=2.0`
- `pytest` for running the smoke test

Install dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
```

Install the package in editable mode:

```bash
python -m pip install -e .
```

## Quick Check

PowerShell:

```powershell
$env:PYTHONPATH="src"
python -m pcos.cli.main smoke --config configs/pcea_medium_cluster15_benchmark.json
python -m pytest -q tests/test_smoke_env.py
```

Bash:

```bash
export PYTHONPATH=src
python -m pcos.cli.main smoke --config configs/pcea_medium_cluster15_benchmark.json
python -m pytest -q tests/test_smoke_env.py
```

The smoke command runs a small scheduling check and prints basic metrics for the
available baseline policies.

## Main Evaluation

The main grouped evaluation submits five independent workflows together to a
shared heterogeneous cluster.

PowerShell:

```powershell
$env:PYTHONPATH="src"
python scripts/evaluate_group5_benchmark.py `
  --config configs/pcea_medium_cluster15_benchmark.json `
  --pcea-config configs/train_pcea_medium_cluster15_benchmark_pcea_anti_idle.json `
  --checkpoint models/pcea_aer/best_energy_pcea_ppo.pt `
  --split benchmark `
  --limit 135 `
  --group-size 5 `
  --output outputs/group5_benchmark
```

Bash:

```bash
export PYTHONPATH=src
python scripts/evaluate_group5_benchmark.py \
  --config configs/pcea_medium_cluster15_benchmark.json \
  --pcea-config configs/train_pcea_medium_cluster15_benchmark_pcea_anti_idle.json \
  --checkpoint models/pcea_aer/best_energy_pcea_ppo.pt \
  --split benchmark \
  --limit 135 \
  --group-size 5 \
  --output outputs/group5_benchmark
```

The precomputed grouped result table is:

```text
results/01_main_group5_workflow_anti_idle_default/summary.csv
```

Main grouped result summary:

| Method | Mean active-server energy |
|---|---:|
| HEFT | 484057.08 |
| PCEA | 411597.17 |
| PCEA-AER | 397300.31 |

In this result table, the mean deadline miss rate is 0 for HEFT, PCEA, and
PCEA-AER.

## Additional Result Tables

Additional compact CSV result tables are included under `results/`:

- `results/02_load_sensitivity_group1_5_10_anti_idle_default/load_summary.csv`
- `results/03_ablation_aer_anti_idle_default/ablation_aer_summary.csv`
- `results/04_pcea_component_ablation/pcea_component_summary.csv`
- `results/05_aer_constraint_ablation/aer_constraint_summary.csv`
- `results/06_power_envelope_sensitivity/power_summary.csv`
- `results/07_idle_timeout_sensitivity/idle_timeout_summary.csv`
- `results/09_runtime_and_migration_analysis_final/runtime_metrics.csv`
- `results/09_runtime_and_migration_analysis_final/migration_type_summary.csv`

## Optional Single-Workflow Evaluation

PowerShell:

```powershell
$env:PYTHONPATH="src"
python scripts/evaluate_pcea_aer.py `
  --config configs/pcea_medium_cluster15_benchmark.json `
  --pcea-config configs/train_pcea_medium_cluster15_benchmark_pcea_anti_idle.json `
  --checkpoint models/pcea_aer/best_energy_pcea_ppo.pt `
  --split benchmark `
  --output outputs/single_workflow_eval
```

Bash:

```bash
export PYTHONPATH=src
python scripts/evaluate_pcea_aer.py \
  --config configs/pcea_medium_cluster15_benchmark.json \
  --pcea-config configs/train_pcea_medium_cluster15_benchmark_pcea_anti_idle.json \
  --checkpoint models/pcea_aer/best_energy_pcea_ppo.pt \
  --split benchmark \
  --output outputs/single_workflow_eval
```

## Optional Dataset Rebuild

The included processed benchmark is sufficient for running the evaluation
scripts.

To rebuild processed workflow files locally, first place the raw Alibaba trace
CSV files under:

```text
datasets/alibaba_raw/batch_task.csv
datasets/alibaba_raw/batch_instance.csv
```

Then run:

```bash
export PYTHONPATH=src
python scripts/build_alibaba_pcea_dataset.py --export-dir datasets/alibaba_pcea/processed
```

Raw Alibaba trace files are not included in this repository.

## Method Summary

The scheduling environment models a heterogeneous CPU-GPU cluster and a
time-varying power envelope.

The evaluation follows two steps:

1. A PPO-based scheduler generates an initial feasible schedule.
2. AER post-processing attempts local task reallocations to reduce active-server
   energy.

A reallocation is accepted only when the replayed schedule satisfies the
scheduler feasibility checks, including resource capacity, DAG precedence,
deadline behavior, and makespan behavior.

## License and Data Terms

The source code is released under the MIT License. See `LICENSE`.

The original Alibaba Cluster Trace 2018 data are not redistributed here and
remain subject to the terms of the original data provider.
