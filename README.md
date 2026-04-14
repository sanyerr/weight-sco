# Weighted Soft Condorcet Optimization

This repository contains the code for the paper **"Weighted Soft Condorcet Optimization"** (GAIW @ AAMAS 2026). Weighted SCO extends Soft Condorcet Optimization (Lanctot et al., 2025) by incorporating Vigna's weighted Kendall-tau distance to prioritize disagreements at the top of the rankings.

## Overview

Standard SCO treats all pairwise disagreements equally. Weighted SCO assigns higher penalties to disagreements at top-ranked positions using three weighting schemes: logarithmic, hyperbolic, and quadratic.

**Key findings:**
* **Logarithmic weights are most effective:** Logarithmic SCO discovers Condorcet winners most frequently on the full PrefLib dataset (94.8%), followed by hyperbolic (93.5%), quadratic (91.1%), and standard SCO (81.1%).
* **Global consistency preserved:** Logarithmic weights maintain near-identical Kemeny-Young performance to standard SCO (71.6% vs 74.0% perfect match), while more aggressive weightings degrade it.
* **Top-k accuracy:** Logarithmic weights consistently achieve the best or tied-best performance across all synthetic metrics, with significant improvements under skill-matched matchups.
* **Theoretical characterization:** The Condorcet guarantee holds when the winner's margin exceeds a threshold determined by the weight ratio; only constant weights give an unconditional guarantee (Theorem 1).

## Requirements & Setup

* **Environment:** Python 3.x, NumPy, pandas, scipy, matplotlib, tqdm, [preflibtools](https://pypi.org/project/preflibtools/).
* **Data:** PrefLib experiments require [PrefLib data](https://www.preflib.org/) placed in `Data/PrefLib-Data-main/`.

---

## Repository Structure

### Core
| File | Description |
| :--- | :--- |
| `sco.py` | Core algorithm: `update_ratings_batch()` implements SGD with weighted gradient accumulation. |
| `loss.py` | Sigmoid loss function and gradient computation for soft Kendall-tau. |
| `ground_truth.py` | Condorcet winner detection, Kemeny-Young optimal ranking (brute force), and evaluation metrics. |
| `experiment.py` | PrefLib file loader and SCO trainer for all four weighting schemes. |

### Experiments
| Paper Section | Script | Description |
| :--- | :--- | :--- |
| 7.1 (PrefLib) | `run_batch.py` | Runs all SCO variants on PrefLib files with $\le 10$ candidates, comparing against Kemeny-Young. |
| 7.1 (Condorcet) | `condorcet_experiment.py` | Tests all PrefLib files (no candidate limit) for Condorcet winner detection. |
| 7.1.1 (Convergence) | `plot_condorcet_convergence.py` | Tracks convergence of the Condorcet winner's rank during optimization. |
| 7.2 (Synthetic) | `synthetic_experiment_merged.py` | Synthetic tournaments with known ground truth ratings (20 agents, uniform and skill-matched). |

### Analysis & Plotting
| To generate... | Run script... | Input |
| :--- | :--- | :--- |
| Table 1 (PrefLib metrics) | `analyze_kemeny-young_results.py` | `replication_results_multi.csv` |
| Condorcet detection rates | `calculate_condorcet_result.py` | `condorcet_efficiency_full.csv` |
| Synthetic tables + significance tests | `analyze_synthetic_results.py` | `synthetic_results_merged.csv` |
| Figure 3 (global metrics plot) | `plot_synthetic_global.py` | `synthetic_results_merged.csv` |
| Figure 4 (top-k metrics plot) | `plot_synthetic_topk.py` | `synthetic_results_merged.csv` |
| Figure 4 compact variant | `plot_synthetic_topk_compact.py` | `synthetic_results_merged.csv` |

---

## Weighting Functions

For rank positions $i < j$ (0-indexed), the weight of an exchange is $w(i) + w(j)$:

* **Logarithmic:** $w_{\text{log}}(i, j) = \frac{1}{\ln(i+e)} + \frac{1}{\ln(j+e)}$
* **Hyperbolic:** $w_{\text{hyp}}(i, j) = \frac{1}{i+1} + \frac{1}{j+1}$
* **Quadratic:** $w_{\text{quad}}(i, j) = \frac{1}{(i+1)^2} + \frac{1}{(j+1)^2}$

Setting $w(i,j) = 1$ for all pairs recovers standard (unweighted) SCO.

## Hyperparameters

Following Lanctot et al. (2025):
* **Learning rate ($\alpha$):** 0.01
* **Temperature ($\tau$):** 1.0
* **Iterations ($T$):** 10,000
* **Rating bounds:** $[0, 100]$ (Initial $\theta = 50.0$)
* **Batch size:** 32 (PrefLib) or 16 (Synthetic)
