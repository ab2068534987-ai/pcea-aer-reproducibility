# PCEA-AER Reproducibility Package

This repository contains a curated reproducibility package for **PCEA-AER**:
**Power-Compute Envelope Advantage PPO with Active-Window Energy Reallocation**.
The code studies heterogeneous CPU-GPU DAG workflow scheduling under a
time-varying power-envelope constraint. A trained PCEA-PPO policy first
generates a feasible schedule, and an inference-time AER step then attempts
local task reallocations to reduce active-server energy.

This repository is not a full development workspace. It keeps the files needed
to inspect the implementation and rerun the reported evaluation, while omitting
raw trace files and intermediate exploratory materials.

## Repository Contents

| Path | Contents |
|---|---|
| `src/pcos/` | Core scheduler, environment, baseline heuristics, PPO policy, and AER implementation |
| `configs/` | Experiment and training configuration files |
| `datasets/alibaba_pcea/processed_benchmark/` | Processed workflow JSON files and manifest CSV files |
| `models/pcea_aer/best_energy_pcea_ppo.pt` | Trained checkpoint used by the evaluation scripts |
| `scripts/` | Dataset construction and evaluation entry points |
| `results/` | CSV result tables reported with this artifact |
| `docs/` | Short method, data, experiment, and result notes |

## Dataset Information

The included benchmark data are derived from Alibaba Cluster Trace 2018. The
raw Alibaba CSV files are not redistributed in this repository and should be
obtained from the original Alibaba Cluster Trace Program subject to its terms
of use.

The processed benchmark root is:

```text
datasets/alibaba_pcea/processed_benchmark/
```

The reported grouped benchmark uses **135 processed workflows**, arranged as
27 groups of 5 workflows. Each workflow is stored as one JSON file. The manifest
CSV files contain:

```text
workflow_path, format, split, repeat, weight, workflow_id
```

Each workflow JSON contains:

- `workflow_id`: workflow identifier.
- `submit_time`: original or derived submit time.
- `tasks`: DAG task records with fields such as `id`, `cpu`, `gpu`, `mem`,
  `base_duration`, `gpu_intensity`, `profile_name`, `preferred_type`, and
  `deadline`.
- `edges`: precedence edges with `src`, `dst`, and `data_mb`.
- `makespan_target`: target completion time used by the scheduler.
- `metadata`: derived DAG properties such as node count, edge count, depth,
  width estimate, total work estimate, and source trace label.

## Code Information

Important implementation files are:

- `src/pcos/env/scheduler_env.py`: scheduling environment and episode metrics.
- `src/pcos/env/power_envelope.py`: time-varying power-envelope model.
- `src/pcos/baselines/heuristics.py`: baseline policies including HEFT and
  minimum-energy heuristics.
- `src/pcos/rl/pcea_ppo.py`: PCEA-PPO policy and training support.
- `src/pcos/postprocess/active_window_energy_reallocation.py`: AER
  post-processing step.
- `scripts/evaluate_pcea_aer.py`: single-workflow evaluation entry point.
- `scripts/evaluate_group5_benchmark.py`: grouped workload evaluation entry
  point.
- `scripts/build_alibaba_pcea_dataset.py`: optional rebuild script for local
  raw Alibaba trace files.

## Requirements

- Python 3.10 or newer.
- Python packages listed in `requirements.txt`:
  - `numpy>=1.22`
  - `pandas>=1.5`
  - `torch>=2.0`
- Optional for the smoke test suite: `pytest`.

PyTorch may be installed as a CPU build or a CUDA build, depending on the local
machine. The included commands below run on CPU by default.

## Installation

From the repository root:

```bash
python -m pip install -e .
```

If dependencies are not already installed:

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
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

The smoke command runs a small scheduler check and prints metrics for the
available baseline policies.

## Reproducing the Main Evaluation

The precomputed result tables are stored under `results/`. To rerun the main
grouped evaluation and write fresh outputs under `outputs/`, use:

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

The main precomputed grouped result is:

```text
results/01_main_group5_workflow_anti_idle_default/summary.csv
```

The grouped result table reports the following mean active-server energy:

| Method | Mean active-server energy |
|---|---:|
| HEFT | 484057.08 |
| PCEA | 411597.17 |
| PCEA-AER | 397300.31 |

In this grouped result, `deadline_miss_rate_mean` is 0 for HEFT, PCEA, and
PCEA-AER.

## Additional Evaluation Tables

Additional compact CSV tables are included for inspection:

- `results/02_load_sensitivity_group1_5_10_anti_idle_default/load_summary.csv`
- `results/03_ablation_aer_anti_idle_default/ablation_aer_summary.csv`
- `results/04_pcea_component_ablation/pcea_component_summary.csv`
- `results/05_aer_constraint_ablation/aer_constraint_summary.csv`
- `results/06_power_envelope_sensitivity/power_summary.csv`
- `results/07_idle_timeout_sensitivity/idle_timeout_summary.csv`
- `results/09_runtime_and_migration_analysis_final/runtime_metrics.csv`
- `results/09_runtime_and_migration_analysis_final/migration_type_summary.csv`

## Optional Dataset Rebuild

The included processed benchmark is sufficient to run the evaluation scripts.
To rebuild processed workflow files locally from raw Alibaba trace CSV files,
place the following files under `datasets/alibaba_raw/`:

```text
datasets/alibaba_raw/batch_task.csv
datasets/alibaba_raw/batch_instance.csv
```

Then run:

```bash
export PYTHONPATH=src
python scripts/build_alibaba_pcea_dataset.py --export-dir datasets/alibaba_pcea/processed
```

The rebuild step is optional and requires the raw trace files, which are not
included in this repository.

## Method Summary

The evaluation uses a heterogeneous cluster configuration
(`cluster_15_realistic`) and a medium peak-valley power-envelope setting.
PCEA-PPO selects scheduling actions using the trained checkpoint in
`models/pcea_aer/`. AER is then applied after the initial schedule. A task
reallocation is accepted only when the replayed schedule preserves the
feasibility checks used by the scheduler, including DAG precedence, resource
capacity, deadline behavior, and makespan behavior.

## Citation

If you use this artifact, cite the archived reproducibility package:

```text
Li Y, Hou W, Ding Z, Long Y, Li G, Wen G. 2026.
PCEA-AER: Power-Compute Envelope Advantage PPO with Active-Window Energy Reallocation.
Zenodo. https://doi.org/10.5281/zenodo.21071243
```

Citation metadata are also provided in `CITATION.cff` and `.zenodo.json`.

## License and Data Terms

The source code in this repository is released under the MIT License; see
`LICENSE`. The original Alibaba Cluster Trace 2018 data are provided by the
Alibaba Cluster Trace Program and remain subject to the original data terms.

## Data and Code Availability

The source code, experimental configurations, processed benchmark files,
training and evaluation scripts, checkpoint, and result tables supporting this
artifact are available at GitHub:

```text
https://github.com/ab2068534987-ai/pcea-aer-reproducibility
```

The archived version corresponding to this manuscript is available at Zenodo:

```text
https://doi.org/10.5281/zenodo.21071243
```
