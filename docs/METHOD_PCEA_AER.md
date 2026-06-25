# PCEA-AER Method Notes

PCEA-AER stands for **Power-Compute Envelope Advantage PPO with Active-Window Energy Reallocation**.

The method has two stages:

1. **PCEA-PPO scheduling** generates a feasible schedule for heterogeneous CPU-GPU DAG workflows under cluster-resource and power-envelope constraints.
2. **Active-Window Energy Reallocation (AER)** is applied after the initial schedule is produced. It attempts local task migrations that reduce active-server energy, especially active idle energy.

AER accepts a migration only when the replayed schedule preserves the feasibility checks used by the scheduler:

- no increase in makespan;
- no increase in deadline miss rate;
- CPU, GPU, memory, and machine-resource feasibility;
- DAG precedence feasibility.

Main implementation locations:

- PPO policy: `src/pcos/rl/pcea_ppo.py`
- Scheduling environment: `src/pcos/env/scheduler_env.py`
- Power-envelope model: `src/pcos/env/power_envelope.py`
- AER module: `src/pcos/postprocess/active_window_energy_reallocation.py`
- Single-workflow evaluation: `scripts/evaluate_pcea_aer.py`
- Group evaluation: `scripts/evaluate_group5_benchmark.py`

The uploaded checkpoint is available at `models/pcea_aer/best_energy_pcea_ppo.pt`.