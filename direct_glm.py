"""Direct GLM fitting on combined AFib data.

This script performs direct logistic regression analysis on AFib data by combining
all data sources and fitting models using statsmodels GLM. This serves as a gold
standard comparison for the federated analysis approach.

The analysis:
1. Loads Cohort_extended.csv and Diagnoses_extended.csv from all data nodes
2. Combines the data into a single dataset
3. Prepares analysis data with the same preprocessing as federated analysis
4. Fits logistic regression models directly using statsmodels
5. Outputs results in the same format as the federated analysis for comparison
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings

from afib_analysis_utils import AFibAnalyzerMixin, MODEL_DEFINITIONS

warnings.filterwarnings("ignore")

__author__ = "Jules Kreuer, jules.kreuer@uni-tuebingen.de"
__version__ = "0.0.0"

COHORT_KEY = "cohort.csv"
DIAGNOSES_KEY = "diagnoses.csv"


class DirectGLMAnalyzer(AFibAnalyzerMixin):
    """Analyzer that loads all data and fits GLMs directly."""

    def __init__(self, data_paths: List[str]):
        self.data_paths = data_paths
        self.cohort_df = None
        self.diagnoses_df = None
        self.analysis_df = None
        self.subcohorts: Dict[str, pd.DataFrame] = {}

    def _load_and_prepare_data(self):
        """Load and combine data from all nodes, then prepare for analysis."""
        if self.analysis_df is not None:
            return

        cohort_dfs = []
        diagnoses_dfs = []

        for path in self.data_paths:
            cohort_path = os.path.join(path, COHORT_KEY)
            diagnoses_path = os.path.join(path, DIAGNOSES_KEY)

            if not os.path.exists(cohort_path) or not os.path.exists(diagnoses_path):
                raise FileNotFoundError(
                    f"Data files not found in {path}. "
                    f"Expected {COHORT_KEY} and {DIAGNOSES_KEY}."
                )

            cohort_dfs.append(pd.read_csv(cohort_path, sep=";", encoding="utf-8"))
            diagnoses_dfs.append(pd.read_csv(diagnoses_path, sep=";", encoding="utf-8"))

        # Combine all data
        self.cohort_df = pd.concat(cohort_dfs, ignore_index=True)
        self.diagnoses_df = pd.concat(diagnoses_dfs, ignore_index=True)

        # Use the same preprocessing as federated analysis
        self._prepare_analysis_data()
        self._filter_data()
        self._prepare_subcohorts()

    def _fit_glm_model(
        self, df: pd.DataFrame, outcome: str, predictors: List[str]
    ) -> Dict[str, Any]:
        """Fit a logistic regression model directly using statsmodels."""
        df_model = df.copy()
        df_model["gender"] = df_model["gender_numeric"]
        df_model = df_model[[outcome] + predictors].dropna()

        if len(df_model) < 10 or df_model[outcome].nunique() < 2:
            return {
                "status": "failed",
                "error": f"Insufficient data: n={len(df_model)}, outcome_unique={df_model[outcome].nunique() if len(df_model) > 0 else 0}",
            }

        y = df_model[outcome]
        X = sm.add_constant(df_model[predictors], prepend=True)

        try:
            # Fit logistic regression using GLM with binomial family
            model = sm.GLM(y, X, family=sm.families.Binomial())
            result = model.fit()

            # Extract coefficients and standard errors
            predictor_names = ["intercept"] + predictors
            coefficients = dict(zip(predictor_names, result.params))
            stderr = dict(zip(predictor_names, result.bse))
            pvalues = dict(zip(predictor_names, result.pvalues))

            # Calculate deviance
            deviance = result.deviance

            return {
                "status": "success",
                "coefficients": coefficients,
                "stderr": stderr,
                "iterations": (
                    result.fit_history["iteration"]
                    if hasattr(result.fit_history, "__getitem__")
                    else None
                ),
                "converged": result.converged,
            }

        except Exception as e:
            return {
                "status": "failed",
                "error": f"{type(e).__name__}: {str(e)}",
            }

    def fit_all_models(self) -> Dict[str, Dict[str, Any]]:
        """Fit all models across all demographic cohorts.

        Returns:
            Nested dict: {cohort_name: {analysis_name: result}}
        """
        self._load_and_prepare_data()

        # Get all demographic cohorts
        demographic_cohorts = self._get_demographic_cohorts()

        all_results = {}
        total_models = len(demographic_cohorts) * len(MODEL_DEFINITIONS)
        print(
            f"\nFitting {len(MODEL_DEFINITIONS)} models across {len(demographic_cohorts)} cohorts ({total_models} total)...\n"
        )

        for cohort_name, cohort_df in demographic_cohorts.items():
            cohort_label = cohort_name.split("_", 1)[1] if "_" in cohort_name else cohort_name
            print(f"Cohort: {cohort_label} (n={len(cohort_df)})")

            if cohort_df.empty:
                all_results[cohort_name] = {
                    "status": "failed",
                    "error": "Cohort is empty",
                    "n_records": 0,
                }
                continue

            # Prepare subcohorts for this demographic cohort
            self._prepare_subcohorts(cohort_df)

            cohort_results = {}
            for name, (outcome, predictors) in MODEL_DEFINITIONS.items():
                subcohort_df = self.subcohorts.get(name)
                if subcohort_df is None or subcohort_df.empty:
                    cohort_results[name] = {
                        "status": "failed",
                        "error": "Subcohort is empty",
                    }
                    continue

                cohort_results[name] = self._fit_glm_model(subcohort_df, outcome, predictors)

                if cohort_results[name]["status"] == "success":
                    print(f"  {name}: SUCCESS")
                else:
                    print(f"  {name}: FAILED - {cohort_results[name]['error']}")

            all_results[cohort_name] = {
                "status": "completed",
                "n_records": len(cohort_df),
                "analyses": cohort_results,
            }
            print()

        return all_results

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the combined dataset."""
        if self.analysis_df is None:
            raise RuntimeError("Data not loaded")

        df = self.analysis_df
        return {
            "total_records": len(df),
            "nt_pro_bnp_mean": float(df["nt_pro_bnp_value"].mean()),
            "nt_pro_bnp_std": float(df["nt_pro_bnp_value"].std()),
            "age_mean": float(df["age"].mean()),
            "age_std": float(df["age"].std()),
            "total_atrial_fibrillation": int(df["AtrialFibrillation"].sum()),
            "total_heart_failure": int(df["HeartFailure"].sum()),
            "total_myocardial_infarction": int(df["MyocardialInfarction"].sum()),
            "total_stroke": int(df["Stroke"].sum()),
            "gender_distribution": df["gender"].value_counts().to_dict(),
            "n_nodes": 1,  # Direct analysis treats all data as single node
        }


def main():
    """Run direct GLM analysis on combined data across all demographic cohorts."""
    node_data_paths = ["data/node_1", "data/node_2"]

    print(f"\n{'='*60}")
    print(f"Direct GLM Analysis")
    print(f"{'='*60}\n")

    analyzer = DirectGLMAnalyzer(node_data_paths)
    cohort_results = analyzer.fit_all_models()
    summary_stats = analyzer.get_summary_stats()

    # Count successful analyses
    total_analyses = 0
    successful_analyses = 0
    for cohort_name, cohort_data in cohort_results.items():
        if "analyses" in cohort_data:
            for analysis_name, analysis_result in cohort_data["analyses"].items():
                total_analyses += 1
                if analysis_result.get("status") == "success":
                    successful_analyses += 1

    # Compile final results
    final_results = {
        "overall_status": "completed",
        "summary": {
            "dataset_statistics": summary_stats,
            "total_cohorts": len(cohort_results),
            "total_analyses": total_analyses,
            "successful_analyses": successful_analyses,
        },
        "cohort_results": cohort_results,
    }

    result_json = json.dumps(final_results, indent=2)
    # print(result_json)

    # Write to output file
    os.makedirs("output", exist_ok=True)
    output_path = "output/direct.json"
    with open(output_path, "w") as f:
        f.write(result_json)
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
