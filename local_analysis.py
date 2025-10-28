"""Local AFib analysis mimicking a federated setup.

This script performs a logistic regression analysis on AFib (Vorhofflimmern/Atrial Fibrillation)
data from multiple local data sources, simulating a federated learning environment without
the need for the FLAME framework. Each data source is treated as a separate node.

The analysis implements an iterative GLM fitter (Iteratively Reweighted Least Squares),
mimicking the approach of DataSHIELD's `ds.glm` function. It is mathematically
equivalent to pooling the data and fitting a single model.

The analysis process:
1.  Initializes an analyzer for each local data node (e.g., 'data/node_1', 'data/node_2').
2.  Each analyzer loads and prepares its respective Cohort_extended.csv and Diagnoses_extended.csv.
3.  An aggregator iteratively coordinates the model fitting:
    a. Analyzers compute score vectors and information matrices based on the current model coefficients.
    b. The aggregator combines these partial results to update the global model coefficients.
4.  This process repeats until the model converges or a maximum number of iterations is reached.
5.  The final aggregated model and summary statistics are printed as a JSON object.
"""

from __future__ import annotations

import traceback
from typing import Any, Dict, List
import os

import pandas as pd
import numpy as np
import warnings

from afib_analysis_utils import (
    AFibAnalyzerMixin,
    AFibAggregatorMixin,
    MAX_ITERATIONS,
    run_analysis_iteration,
    run_aggregation_iteration,
)

warnings.filterwarnings("ignore")

__author__ = "Jules Kreuer, jules.kreuer@uni-tuebingen.de"
__version__ = "0.0.0"

COHORT_KEY = "Cohort_extended.csv"
DIAGNOSES_KEY = "Diagnoses_extended.csv"


class AFibAnalyzer(AFibAnalyzerMixin):
    """Analyzer that performs one round of iterative GLM fitting on local data."""

    def __init__(self, node_id: str, data_path: str):
        self.node_id = node_id
        self.data_path = data_path
        self.cohort_df = None
        self.diagnoses_df = None
        self.analysis_df = None
        self.subcohorts: Dict[str, pd.DataFrame] = {}

    def _load_and_prepare_data(self):
        """Load, prepare, and cache the analysis dataframe and subcohorts from local files."""
        if self.analysis_df is not None:
            return

        cohort_path = os.path.join(self.data_path, COHORT_KEY)
        diagnoses_path = os.path.join(self.data_path, DIAGNOSES_KEY)

        if not os.path.exists(cohort_path) or not os.path.exists(diagnoses_path):
            raise FileNotFoundError(
                f"Data files not found in {self.data_path}. "
                f"Expected {COHORT_KEY} and {DIAGNOSES_KEY}."
            )

        self.cohort_df = pd.read_csv(cohort_path, sep=";", encoding="utf-8")
        self.diagnoses_df = pd.read_csv(diagnoses_path, sep=";", encoding="utf-8")

        self._prepare_analysis_data()
        self._filter_data()
        self._prepare_subcohorts()

    def analysis_method(
        self,
        aggregator_results: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        """Perform one round of federated GLM fitting."""
        # Load and prepare data (cached after first call)
        self._load_and_prepare_data()

        return run_analysis_iteration(self, aggregator_results)


class AFibAggregator(AFibAggregatorMixin):
    """Aggregator that manages iterative GLM fitting."""

    def __init__(self):
        self.model_states = {}  # Stores beta, deviance, convergence status for each model
        self.summary_stats = {}
        self.iteration = 0

    def analysis_method(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results and perform one IRLS update."""
        return run_aggregation_iteration(self, analysis_results)

    def has_converged(self, num_iterations: int) -> bool:
        """Check if all models have converged or max iterations reached."""
        if num_iterations >= MAX_ITERATIONS:
            return True

        if num_iterations == 0:
            return False  # Always run at least one iteration

        if not self.model_states:
            return True  # Stop if no models are running after initialization

        return all(state.get("converged", False) for state in self.model_states.values())


def main():
    """Configure and run the local federated AFib analysis."""
    node_data_paths = ["data/node_1", "data/node_2"]
    analyzers = [AFibAnalyzer(f"node_{i+1}", path) for i, path in enumerate(node_data_paths)]
    aggregator = AFibAggregator()

    print(f"\n{'='*60}")
    print(f"Federated AFib Analysis ({len(analyzers)} nodes)")
    print(f"{'='*60}")

    aggregator_results = None
    num_iterations = 0

    while not aggregator.has_converged(num_iterations):
        num_iterations += 1
        analyzer_results = [analyzer.analysis_method(aggregator_results) for analyzer in analyzers]

        # Check for errors from analyzers
        for res in analyzer_results:
            if res["status"] == "error":
                print(f"ERROR in {res['node_id']}: {res['error']}")
                if "traceback" in res:
                    print(res["traceback"])
                return

        aggregator_results = aggregator.analysis_method(analyzer_results)

        if "error" in aggregator_results:
            print(f"ERROR: {aggregator_results['error']}")
            break

    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*60}\n")

    result_json = aggregator.get_result()
    print(result_json)

    # Write to output file
    os.makedirs("output", exist_ok=True)
    output_path = "output/local.json"
    with open(output_path, "w") as f:
        f.write(result_json)
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
