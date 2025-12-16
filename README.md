# FLAME-AFib Analysis

This repository contains three implementations of atrial fibrillation (AFib) analysis using generalized linear models (GLM):

## Analysis Versions
All three versions use functions from `afib_analysis_utils.py` to ensure implementation equivalence across methods.

### 1. Federated Analysis (`federated_analysis.py`)
Runs on the PrivateAIM Platform (FLAME). This analysis shall not be executed on real data as we have explicit logging and tracebacks that are shared to the Hub.

**Use:**
- Project: AFib-test
- Nodes: 1, 2, and 6
- Upload: federated_analysis.py (entry point) and afib_analysis_utils.py
- Docker Image: python/use-cases fedstats

### 2. Local Analysis (`local_analysis.py`)
Simulates the federated setting locally for development and testing. Mimics the federated workflow by processing data from multiple nodes sequentially on a single machine.

### 3. Direct Analysis (`direct_glm.py`)
Performs standard GLM fitting on combined data as a reference implementation. This represents the gold standard result when all data is pooled centrally.

## Privacy

This branch implements an output-level differential privacy (DP) variant of the
AFib analysis. The core preprocessing and statistical models are unchanged;
only the **final aggregated results** are perturbed.

We assume a trusted execution environment for all node-local computation and
federated message passing. Differential privacy is applied exclusively to the
final JSON outputs produced by the direct (pooled), local-federated, and FLAME
analyses.

To obtain finite, data-independent sensitivities, inputs are bounded via
deterministic clipping and very small cohorts are suppressed rather than
reported. On top of these bounded aggregates, we apply independent Laplace
mechanisms (via OpenDP) to each scalar summary and model statistic, using
per-statistic global sensitivities and a common privacy parameter ε. This
provides ε-DP protection for each individual reported statistic; the joint
release of all statistics should be interpreted under standard DP composition
for this one-shot batch report.

Logistic regression outputs (coefficients and standard errors) are additionally
perturbed to reduce disclosure risk, but the underlying fitting procedure is
not itself differentially private. All DP parameters (clipping bounds,
sensitivities, and ε configuration) are defined explicitly in the code.

## Validation

`compare_results.py` compares outputs from local and direct analyses to validate the federated implementation's correctness.


## Data

Example data in `data/node_1/` and `data/node_2/` has been extended with random values to cover all analysis conditions. This data is already uploaded to the MinIO buckets of the corresponding nodes on the PrivateAIM platform.

## Requirements

Install dependencies with:
```
pip install -r requirements.txt
```