# Simulations of Particulate Flows — Project 3: Particulate-Bed Digital Twin (EnKF + Learned Drag)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](#) [![PyTorch](https://img.shields.io/badge/PyTorch-lightgrey.svg)](#) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#)

**GitHub:** https://github.com/Riya2382


# Particulate Bed Digital Twin (Toy) — EnKF + Learned Drag Correction

**Goal:** Minimal 1D dynamic model of a bubbling/packed bed with gas flow. We simulate a "true" process, assimilate noisy sensor data with an Ensemble Kalman Filter (EnKF), and *learn a residual drag correction* that improves forecasts. This mimics physics+ML hybrid twins for real‑time control.

## Components
- `src/sim_truth.py`: synthesizes ground‑truth trajectories with time‑varying gas flow rate and a nonlinear drag law.
- `src/enkf_twin.py`: twin model with EnKF data assimilation.
- `src/learn_drag.py`: tiny NN learns a residual correction ΔC_d from history to reduce forecast error.
- Outputs: plots + `.npz` artifacts in `outputs/`.

## Run
```bash
python src/sim_truth.py
python src/enkf_twin.py --T 200
python src/learn_drag.py --epochs 5
```
You’ll get before/after plots showing improved tracking and short‑horizon prediction.

## Why this helps
- Shows **digital‑twin thinking**: grey‑box physics, online estimation, learned closures.
- Clean baseline others can reproduce and evaluate (RMSE, horizon tests).
- Natural extension to CFD‑DEM: replace 1D toy with reduced models or coarse CFD states.
## How this maps to moving/fluidized beds & rotary kilns

- Mirrors a **grey‑box twin**: simple physics core + **EnKF** assimilation + a small **learned residual** to correct model inadequacies (e.g., drag correlations).
- Directly applicable to fluidized beds and rotary kilns with limited sensors (bed height, ΔP, temperature): fuses data and physics for **robust short‑horizon forecasts**.
- Ready to extend with multi-sensor streams and actuator inputs for **model predictive control**.


> **Use in your email:** Include one of the generated plots and a 1–2 line summary:
> *“Built a small, reproducible demo aligning with recurrence/operator-learning/digital-twin ideas and showed real-time rollouts/forecasting on toy data; ready to swap in CFD/CFD‑DEM snapshots.”*
