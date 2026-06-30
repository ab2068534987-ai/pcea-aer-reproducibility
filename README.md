# PCEA-AER Reproducibility Package

This repository contains the reproducibility package for **PCEA-AER**: **Power-Compute Envelope Advantage PPO with Active-Window Energy Reallocation**. PCEA-AER schedules heterogeneous CPU-GPU DAG workflows under a time-varying power-envelope constraint and then applies an inference-time active-window reallocation step to reduce active-server energy while preserving schedule feasibility.

## Repository Scope

This is a curated artifact repository, not a full working directory dump. It includes the material needed to inspect and reproduce the reported evaluation:

- Core implementation under `src/pcos/`
- Experiment configurations under `configs/`
- Processed benchmark data under `datasets/alibaba_pcea/processed_benchmark/`
- Trained checkpoint under `models/pcea_aer/best_energy_pcea_ppo.pt`
- Main evaluation scripts under `scripts/`
- Result tables under `results/`
- Method, dataset, and experiment notes under `docs/`

The raw Alibaba trace CSV files are not redistributed. The repository includes the processed benchmark split used by the experiments.

## Installation

```bash
python -m pip install -e .
```

PyTorch is required for PPO-based evaluation. Install the CPU or CUDA build that matches your environment.

## Smoke Test

```powershell
$env:PYTHONPATH="src"
python -m pcos.cli.main smoke --config configs/pcea_medium_cluster15_benchmark.json
```

## Main Evaluations

Single-workflow benchmark:

```powershell
$env:PYTHONPATH="src"
python scripts/evaluate_pcea_aer.py
```

Group workload benchmark with five workflows submitted together:

```powershell
$env:PYTHONPATH="src"
python scripts/evaluate_group5_benchmark.py
```

Both scripts use the processed benchmark data and the checkpoint in `models/pcea_aer/` by default.

## Key Results

| Scenario | HEFT energy | PCEA energy | PCEA-AER energy | PCEA-AER vs. HEFT |
|---|---:|---:|---:|---:|
| Single workflow | 70037.57 | 70008.89 | 68951.03 | -1.55% |
| Group size 5 | 484057.08 | 411597.17 | 397300.31 | -17.92% |

PCEA-AER keeps `deadline_miss_rate = 0` in the reported evaluation. In the group-size-5 scenario, active-server utilization improves from 0.66 for HEFT to 0.78 for PCEA-AER.

## Data and Results

- Processed benchmark root: `datasets/alibaba_pcea/processed_benchmark/`
- Main group result: `results/01_main_group5_workflow_anti_idle_default/summary.csv`
- Load sensitivity result: `results/02_load_sensitivity_group1_5_10_anti_idle_default/load_summary.csv`
- Evaluation checkpoint: `models/pcea_aer/best_energy_pcea_ppo.pt`

See `docs/` for additional method, dataset, and experiment notes.

## Citation

This repository is prepared for archival on Zenodo. After publishing the `v1.0.0` GitHub release and Zenodo has generated the DOI, replace `10.5281/zenodo.xxxxxxx` with the Version DOI.

```text
Li Y, Hou W, Ding Z, Long Y, Li G, Wen G. 2026. PCEA-AER: Power-Compute Envelope Advantage PPO with Active-Window Energy Reallocation. Zenodo. https://doi.org/10.5281/zenodo.xxxxxxx
```

The citation metadata is available in `CITATION.cff` and `.zenodo.json`.

## Data and Code Availability

The source code, experimental configurations, processed benchmark descriptions, trained checkpoint, evaluation scripts, and result tables supporting this study are available at GitHub: https://github.com/ab2068534987-ai/pcea-aer-reproducibility. The archived version will be available at Zenodo after the GitHub release DOI is generated.
