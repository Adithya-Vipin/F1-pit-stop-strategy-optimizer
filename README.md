# F1 Pit-Stop Strategy Optimizer

An ML-based simulation system that models tyre degradation, predicts lap time evolution, and optimizes Formula 1 pit-stop strategies using Monte Carlo simulation and a policy-based decision engine.

## Overview

Pit-stop timing is one of the most consequential decisions in a Formula 1 race. This project uses historical F1 race data to model how tyre degradation affects lap times, then simulates thousands of strategic scenarios to find the optimal pit window — for both one-stop and two-stop strategies.

## Pipeline

The project is structured as 5 sequential notebooks:

| Notebook | Description |
|---|---|
| `F1_01_dataloading.ipynb` | Load and inspect raw CSVs — 589k lap time records across 1,125 races |
| `F1_02_eda_lap_times.ipynb` | EDA — visualize lap time evolution, mark pit stop laps on time series |
| `F1_03_feature_engineering.ipynb` | Engineer tyre age feature, clean pit laps, build stint-level dataset |
| `F1_04_lap_time_model.ipynb` | Train Linear Regression model: tyre age → lap time; evaluate on test set |
| `F1_05_strategy_simulation.ipynb` | Full strategy simulation — one-stop, two-stop, Monte Carlo, policy engine |

## How It Works

**1. Tyre Degradation Modeling**
Lap times increase as tyre age grows. A Linear Regression model is fit on stint-level data (pit laps excluded) to learn the degradation rate per lap. A physics-informed minimum degradation floor (`0.05s/lap`) is applied to stabilize predictions on noisy data.

**2. Compound Profiles**
Three tyre compounds are modeled with distinct base offset and degradation rates:

| Compound | Base Offset | Degradation/lap |
|---|---|---|
| Soft | 0.0s | 0.18s |
| Medium | +0.4s | 0.10s |
| Hard | +0.8s | 0.06s |

**3. One-Stop Optimizer**
Simulates every candidate pit lap (lap 5 to lap N-5), calculates total race time including a 20s pit time penalty, and finds the optimal pit window that minimizes total race time.

**4. Two-Stop Brute-Force**
Exhaustively evaluates all valid (pit1, pit2) combinations across all compound permutations (Soft/Medium/Hard × 3 stints) to find the globally optimal two-stop strategy.

**5. Monte Carlo Simulation**
Adds lap-time noise (`N(0, 0.3s)`) to each simulated lap and runs 100 stochastic trials per strategy to evaluate robustness under race uncertainty. Reports mean and standard deviation of total race time per strategy.

**6. Policy-Based Pit Decision Engine**
A rule-based policy engine dynamically decides pit timing at each lap based on:
- Current tyre age and degradation rate
- Remaining laps in the race
- Future time loss estimate vs. pit time cost (`future_loss = degradation × tyre_age × remaining_laps`)

If `future_loss > PIT_TIME`, the policy triggers a pit stop. Policy race time closely matches the Monte Carlo optimum — validating the approach and bridging toward RL-style control.

## Dataset

Historical F1 data from the [Ergast Motor Racing API](http://ergast.com/mrd/) / [Kaggle F1 dataset](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020):

- `lap_times.csv` — 589,081 lap records
- `pit_stops.csv` — 11,371 pit stop records
- `races.csv` — 1,125 races (2009–present)
- `circuits.csv` — circuit metadata

> Data files are not included in this repo due to size. Download from Kaggle and place in `data/raw/`.

## Tech Stack

![Python](https://img.shields.io/badge/Python-2b0a3d?style=for-the-badge&logo=python&logoColor=A371F7)
![Pandas](https://img.shields.io/badge/Pandas-2b0a3d?style=for-the-badge&logo=pandas&logoColor=A371F7)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-2b0a3d?style=for-the-badge&logo=scikit-learn&logoColor=A371F7)
![NumPy](https://img.shields.io/badge/NumPy-2b0a3d?style=for-the-badge&logo=numpy&logoColor=A371F7)
![Matplotlib](https://img.shields.io/badge/Matplotlib-2b0a3d?style=for-the-badge&logo=python&logoColor=A371F7)

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib
```

Download the F1 dataset from Kaggle, place CSVs in `data/raw/`, then run notebooks in order (01 → 05). Notebooks were developed in Google Colab — the Drive mount cell can be removed if running locally.

## Key Insight

> The dataset contained limited continuous lap sequences per driver, making pure ML-based degradation modeling noisy. Introducing physics-informed constraints (minimum degradation floor, compound profiles) stabilized the strategy optimization and produced realistic pit windows.

---

*B.M.S College of Engineering | AI & ML Department*
