# Weighted Soft Condorcet Optimization

This repository contains the code for the paper **"Weighted Soft Condorcet Optimization."** Weighted SCO extends Soft Condorcet Optimization (Lanctot et al., 2025) by incorporating weights that prioritize disagreements at the top of the rankings.

## Overview

While standard SCO treats all pairwise disagreements equally, Weighted SCO assigns higher penalties to disagreements at top-ranked positions using hyperbolic or quadratic weighting functions.

**Key findings:**
* **Condorcet Efficiency:** Hyperbolic weighted SCO discovers Condorcet winners more frequently than unweighted SCO (93.4% vs 81.9%).
* **Top-k Accuracy:** Shows modest improvements when ground truth is available.
* **The Trade-off:** Weighted methods naturally perform worse on Kemeny-Young metrics, which are unweighted by definition.

## Requirements & setup

* **Environment:** Python 3.x, NumPy, pandas, tqdm, [preflibtools](https://pypi.org/project/preflibtools/).
* **Data:** Real-world experiments require [PrefLib data](https://www.preflib.org/).

---

## Repository structure

### 1. Core
| File | Description |
| :--- | :--- |
| `sco.py` | Core algorithm: `update_ratings_batch()` implements SGD with weighted gradient accumulation. |
| `loss.py` | Sigmoid loss function and gradient computation for soft Kendall-tau. |

### 2. Experiments
| Section | Script | Description |
| :--- | :--- | :--- |
| **6.1 (PrefLib)** | `run_batch.py` | Runs uniform/hyperbolic/quadratic SCO on files with $\le 10$ candidates. |
| **6.2 (Synthetic)** | `synthetic_experiment_merged.py` | Tests tournaments with known ground truth ratings. |
| **Condorcet winners** | `condorcet_experiment.py` | Tests all PrefLib files (no candidate limit) for Condorcet winner detection. |

### 3. Analysis & Tables
| To generate... | Run script... | Input data source |
| :--- | :--- | :--- |
| **Table 1** | `analyze_kemeny-young_results.py` | `replication_results_multi.csv` |
| **Tables 2-3** | `analyze_synthetic_results.py` | `synthetic_results_merged.csv` |
| **Win Rates** | `calculate_condorcet_result.py` | `condorcet_efficiency_full.csv` |

---

## Weighting functions

For rank positions $i < j$ (0-indexed), we implement the following weights:

* **Hyperbolic (Vigna):** $w_{\text{hyp}}(i, j) = \frac{1}{i+1} + \frac{1}{j+1}$
* **Quadratic:** $w_{\text{quad}}(i, j) = \frac{1}{(i+1)^2} + \frac{1}{(j+1)^2}$

## Hyperparameters

To ensure parity with Lanctot et al. (2025), we use:
* **Learning rate ($\alpha$):** 0.01
* **Temperature ($\tau$):** 1.0
* **Iterations ($T$):** 10,000
* **Rating bounds:** $[0, 100]$ (Initial $\theta = 50.0$)
* **Batch size:** 32 (PrefLib) or 16 (Synthetic)
