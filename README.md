# FLAME-AFib Analysis

Maintainer: [Jules Kreuer](https://github.com/not-a-feature)

This repository contains three implementations of atrial fibrillation (AFib) analysis using generalized linear models (GLM):

## Analysis Versions
All three versions use functions from `afib_analysis_utils.py` to ensure implementation equivalence across methods.

### 1. Federated Analysis (`federated_analysis.py`)
Runs on the PrivateAIM Platform (FLAME). This analysis shall not be executed on real data as we have explicit logging and tracebacks that are shared to the Hub.

**Use:**
- Project: AFib-test
- Nodes: Two Nodes, One Aggregator
- Upload: federated_analysis.py (entry point) and afib_analysis_utils.py
- Docker Image: python/use-cases fedstats
- Check/Conntect the Datastores

### 2. Local Analysis (`local_analysis.py`)
Simulates the federated setting locally for development and testing. Mimics the federated workflow by processing data from multiple nodes sequentially on a single machine.

### 3. Direct Analysis (`direct_glm.py`)
Performs standard GLM fitting on combined data as a reference implementation. This represents the gold standard result when all data is pooled centrally.

## Validation

`compare_results.py` compares outputs from local and direct analyses to validate the federated implementation's correctness.


## Data

Example data in `data/node_1/` and `data/node_2/` has been extended with random values to cover all analysis conditions. This data is already uploaded to the MinIO buckets of the corresponding nodes on the PrivateAIM platform.

## Requirements

Install dependencies with:
```
pip install -r requirements.txt
```
