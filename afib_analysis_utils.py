"""Shared utilities for AFib federated and local analysis.

This module contains common functions and classes used in both federated (FLAME-based)
and local AFib (Vorhofflimmern/Atrial Fibrillation) analysis implementations.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd
import numpy as np
import statsmodels.api as sm

__author__ = "Jules Kreuer, jules.kreuer@uni-tuebingen.de"
__version__ = "0.0.0"

# Analysis parameters
MAX_ITERATIONS = 50
CONVERGENCE_THRESHOLD = 1e-8


class AFibAnalyzerMixin:
    """Mixin providing shared analysis methods for AFib analyzers."""

    def _prepare_analysis_data(self):
        """Prepare analysis dataframe with relevant columns and diagnosis flags."""
        if self.cohort_df is None or self.diagnoses_df is None:
            raise RuntimeError("Cohort or diagnoses DataFrame not loaded")

        # Create pivot of diagnoses to get one row per subject-encounter
        diagnosis_pivot = self.diagnoses_df.pivot_table(
            index=["subject", "encounter.id"],
            columns="code",
            aggfunc="size",
            fill_value=0,
        ).reset_index()

        # Merge with cohort data
        self.analysis_df = self.cohort_df.merge(
            diagnosis_pivot, on=["subject", "encounter.id"], how="left"
        )

        # Fill NaN diagnosis flags with 0
        diagnosis_cols = [
            col for col in self.analysis_df.columns if col.startswith("I") or col.startswith("E")
        ]
        self.analysis_df[diagnosis_cols] = self.analysis_df[diagnosis_cols].fillna(0)

        # Create binary diagnosis flags
        def has_diagnosis_pattern(pattern):
            """Check if any diagnosis column matches the pattern."""
            df = self.analysis_df
            if df is None:
                raise RuntimeError("Analysis DataFrame not prepared")
            matching_cols = [col for col in df.columns if any(p in col for p in pattern)]
            if matching_cols:
                return (df[matching_cols].sum(axis=1) > 0).astype(int)
            return pd.Series([0] * len(df), index=df.index)

        # Diagnosis flags
        self.analysis_df["IdiopathicHypotension"] = has_diagnosis_pattern(["I95.0"])
        self.analysis_df["AtrialFibrillation"] = has_diagnosis_pattern(
            [
                "I48.0",
                "I48.1",
                "I48.2",
                "I48.9",
            ]
        )
        self.analysis_df["MyocardialInfarction"] = has_diagnosis_pattern(
            [
                "I21",
                "I22",
                "I25.2",
            ]
        )
        self.analysis_df["HeartFailure"] = has_diagnosis_pattern(["I50"])
        self.analysis_df["Stroke"] = has_diagnosis_pattern(
            [
                "I60",
                "I61",
                "I62",
                "I63",
                "I64",
                "I69",
            ]
        )

        # Convert NT-proBNP value to numeric and apply unit conversion
        self.analysis_df["nt_pro_bnp_value"] = pd.to_numeric(
            self.analysis_df["NTproBNP.valueQuantity.value"],
            errors="coerce",
        )

        # Apply comparator logic (match R implementation)
        # If comparator is ">", value = value + 1
        # If comparator is "<", value = value - 1
        # TODO Can we remove this?
        if "NTproBNP.valueQuantity.comparator" in self.analysis_df.columns:
            comparator = self.analysis_df["NTproBNP.valueQuantity.comparator"]
            # Ensure we only modify where we have valid values
            mask_greater = comparator == ">"
            mask_less = comparator == "<"

            self.analysis_df.loc[mask_greater, "nt_pro_bnp_value"] += 1
            self.analysis_df.loc[mask_less, "nt_pro_bnp_value"] -= 1

        # Apply unit conversion
        self._convert_nt_pro_bnp_units()

    def _convert_nt_pro_bnp_units(self):
        """Convert NT-proBNP values to uniform unit pg/mL."""
        if self.analysis_df is None:
            raise RuntimeError("Analysis DataFrame not prepared")

        # Ensure required columns exist
        if "NTproBNP.unit" not in self.analysis_df.columns:
            self.analysis_df["NTproBNP.unit"] = np.nan
        if "NTproBNP.unitLabel" not in self.analysis_df.columns:
            self.analysis_df["NTproBNP.unitLabel"] = np.nan

        unit_col = self.analysis_df["NTproBNP.unit"]
        unit_label_col = self.analysis_df["NTproBNP.unitLabel"]

        # R-like behavior: all NA -> copy full unitLabel; else fill NA with unitLabel
        if unit_col.isna().all():
            self.analysis_df["NTproBNP.unit"] = unit_label_col
        else:
            self.analysis_df["NTproBNP.unit"] = unit_col.fillna(unit_label_col)

        self.analysis_df.dropna(subset=["NTproBNP.unit"], inplace=True)

        unit_factors = {
            "pg/ml": 1.0,  # default
            "ng/l": 1.0,
            "pg/dl": 0.01,
            "pg/100ml": 0.01,
            "pg%": 0.01,
            "pg/l": 0.001,
            "pmol/l": 8.457,  # See: https://journals.sagepub.com/doi/full/10.1258/acb.2007.007069 "NT-proBNP concentrations are expressed in picomoles/litre (for conversion to picograms/millilitre they are multiplied by 8.457)."
        }

        unit_lower = self.analysis_df["NTproBNP.unit"].str.lower()
        valid_mask = unit_lower.isin(unit_factors.keys())
        self.analysis_df = self.analysis_df[valid_mask].copy()

        conversion_factors = self.analysis_df["NTproBNP.unit"].str.lower().map(unit_factors)
        self.analysis_df["nt_pro_bnp_value"] *= conversion_factors

        self.analysis_df["NTproBNP.unit"] = "pg/mL"
        self.analysis_df["NTproBNP.unitLabel"] = "picogram per milliliter"

    def _filter_data(self):
        """Filter out missing values."""
        if self.analysis_df is None:
            raise RuntimeError("Analysis DataFrame not prepared")
        self.analysis_df = self.analysis_df[
            (self.analysis_df["nt_pro_bnp_value"].notna())
            & (self.analysis_df["nt_pro_bnp_value"] >= 0)
            & (self.analysis_df["gender"].notna())
            & (self.analysis_df["age"].notna())
        ].copy()

    def _get_demographic_cohorts(self) -> Dict[str, pd.DataFrame]:
        """Create demographic cohorts matching the R implementation.

        Returns:
            Dictionary mapping cohort names to filtered DataFrames
        """
        if self.analysis_df is None:
            raise RuntimeError("Analysis DataFrame not prepared")

        df = self.analysis_df

        cohorts = {
            "01_FullCohort": df.copy(),
            "02_Males": df[df["gender"] == "male"].copy(),
            "03_Females": df[df["gender"] == "female"].copy(),
            "04_Age_18_50": df[(df["age"] >= 18) & (df["age"] <= 50)].copy(),
            "05_Age_51_80": df[(df["age"] >= 51) & (df["age"] <= 80)].copy(),
            "06_Age_81_Inf": df[df["age"] > 80].copy(),
            "07_Males_Age_18_50": df[
                (df["gender"] == "male") & (df["age"] >= 18) & (df["age"] <= 50)
            ].copy(),
            "08_Males_Age_51_80": df[
                (df["gender"] == "male") & (df["age"] >= 51) & (df["age"] <= 80)
            ].copy(),
            "09_Males_Age_81_Inf": df[(df["gender"] == "male") & (df["age"] > 80)].copy(),
            "10_Females_Age_18_50": df[
                (df["gender"] == "female") & (df["age"] >= 18) & (df["age"] <= 50)
            ].copy(),
            "11_Females_Age_51_80": df[
                (df["gender"] == "female") & (df["age"] >= 51) & (df["age"] <= 80)
            ].copy(),
            "12_Females_Age_81_Inf": df[(df["gender"] == "female") & (df["age"] > 80)].copy(),
        }

        return cohorts

    def _prepare_subcohorts(self, cohort_df: pd.DataFrame | None = None):
        """Create and cache all subcohort dataframes for analysis.

        Args:
            cohort_df: Optional DataFrame to use as base cohort. If None, uses self.analysis_df
        """
        if cohort_df is None:
            if self.analysis_df is None:
                raise RuntimeError("Analysis DataFrame not prepared")
            cohort_df = self.analysis_df

        # Recode gender once
        gender_map = {"male": 0, "female": 1}
        cohort_df["gender_numeric"] = cohort_df["gender"].map(gender_map).fillna(2).astype(int)

        # Subcohorts should NOT be filtered by outcome variable
        # They represent different populations where we model the outcome
        self.subcohorts = {
            "CONDITIONTEST": cohort_df.copy(),
            "CONDITION_AFIB": cohort_df.copy(),
            "CONDITION_HIS": cohort_df.copy(),
            "CONDITION_AFIB2": cohort_df[
                (cohort_df["MyocardialInfarction"] == 0) & (cohort_df["Stroke"] == 0)
            ].copy(),
            "CONDITION_HIS2": cohort_df[
                (cohort_df["MyocardialInfarction"] == 0) & (cohort_df["Stroke"] == 0)
            ].copy(),
            "CONDITION_AFIB3": cohort_df[
                (cohort_df["MyocardialInfarction"] == 0)
                & (cohort_df["Stroke"] == 0)
                & (cohort_df["HeartFailure"] == 0)
            ].copy(),
        }

    def _calculate_glm_iteration(
        self,
        df: pd.DataFrame,
        outcome: str,
        predictors: List[str],
        beta: Dict[str, float],
    ) -> Dict[str, Any] | None:
        """Calculate score vector and info matrix for one IRLS iteration.

        Args:
            df: DataFrame with data
            outcome: Outcome variable name
            predictors: List of predictors to use in the model
            beta: Current beta coefficients (dict or array)
        """
        df_model = df.copy()
        df_model["gender"] = df_model["gender_numeric"]
        df_model = df_model[[outcome] + predictors].dropna()

        if len(df_model) < 10 or df_model[outcome].nunique() < 2:
            return None

        y = df_model[outcome].to_numpy()
        X_df = sm.add_constant(df_model[predictors], prepend=True)
        # Use explicit cast to avoid type checker confusion with statsmodels return types
        X = np.asarray(X_df)

        # Convert beta to array in correct order
        # Use 'const' to match statsmodels naming convention
        predictor_names = ["const"] + predictors
        beta_array = np.array([beta[pred] for pred in predictor_names])

        model = sm.GLM(y, X, family=sm.families.Binomial())

        # Calculate score and hessian at current beta
        score_vector = model.score(beta_array)
        hessian = model.hessian(beta_array)

        # Fisher Information Matrix is -Hessian for canonical link (Logit)
        info_matrix = -hessian

        # Calculate deviance
        mu = model.predict(beta_array)
        deviance = model.family.deviance(y, mu)

        return {
            "score_vector": score_vector.tolist(),
            "info_matrix": info_matrix.tolist(),
            "n_samples": len(df_model),
            "deviance": deviance,
            "predictors": predictors,
        }

    def _get_summary_stats(self):
        """Get summary statistics for the node (run once)."""
        if self.analysis_df is None:
            raise RuntimeError("Analysis DataFrame not prepared")
        return {
            "n_total": len(self.analysis_df),
            "nt_pro_bnp_mean": float(self.analysis_df["nt_pro_bnp_value"].mean()),
            "age_mean": float(self.analysis_df["age"].mean()),
            "atrial_fibrillation_count": int(self.analysis_df["AtrialFibrillation"].sum()),
            "heart_failure_count": int(self.analysis_df["HeartFailure"].sum()),
        }


class AFibAggregatorMixin:
    """Mixin providing shared aggregation methods for AFib aggregators."""

    model_states: Dict[str, Dict[str, Any]]
    summary_stats: Dict[str, Any]
    cohort_n_records: Dict[str, int]

    def _initialize_model_states(self, analysis_results: List[Dict[str, Any]]):
        """Initialize state for all models across all cohorts based on the first results."""
        if self.model_states:
            return

        first_result = next((r for r in analysis_results if r["status"] == "success"), None)
        if not first_result:
            return

        # Initialize cohort record counts tracking
        if not hasattr(self, "cohort_n_records"):
            self.cohort_n_records = {}

        # Initialize nested structure: {cohort_name: {model_name: state}}
        first_cohorts = first_result["cohort_results"]
        for cohort_name in first_cohorts.keys():
            self.model_states[cohort_name] = {}
            model_defs = get_model_definitions(cohort_name)
            for name, (outcome, predictors) in model_defs.items():
                self.model_states[cohort_name][name] = {
                    "beta": {},  # Will store {predictor: value} dict
                    "deviance": float("inf"),
                    "converged": False,
                    "iteration": 0,
                    "predictors": predictors,
                    "effective_predictors": None,  # Will be set during first iteration
                }

    def _aggregate_summary_stats(self, analysis_results: List[Dict[str, Any]]):
        """Aggregate overall dataset statistics across nodes."""
        if self.summary_stats:
            return

        successful_results = [
            r
            for r in analysis_results
            if r["status"] == "success" and "summary_stats" in r and r["summary_stats"]
        ]
        if not successful_results:
            return

        # Aggregate overall statistics across all nodes
        total_n = sum(r["summary_stats"]["n_total"] for r in successful_results)

        if total_n == 0:
            self.summary_stats = {
                "total_records": 0,
                "n_nodes": len(successful_results),
            }
            return

        # Compute weighted means and totals
        self.summary_stats = {
            "total_records": total_n,
            "nt_pro_bnp_mean": sum(
                r["summary_stats"]["n_total"] * r["summary_stats"]["nt_pro_bnp_mean"]
                for r in successful_results
            )
            / total_n,
            "nt_pro_bnp_std": self._aggregate_std(
                successful_results, "nt_pro_bnp_mean", "nt_pro_bnp_std", total_n
            ),
            "age_mean": sum(
                r["summary_stats"]["n_total"] * r["summary_stats"]["age_mean"]
                for r in successful_results
            )
            / total_n,
            "age_std": self._aggregate_std(successful_results, "age_mean", "age_std", total_n),
            "total_atrial_fibrillation": sum(
                r["summary_stats"]["atrial_fibrillation_count"] for r in successful_results
            ),
            "total_heart_failure": sum(
                r["summary_stats"]["heart_failure_count"] for r in successful_results
            ),
            "total_myocardial_infarction": sum(
                r["summary_stats"]["myocardial_infarction_count"] for r in successful_results
            ),
            "total_stroke": sum(r["summary_stats"]["stroke_count"] for r in successful_results),
            "gender_distribution": self._aggregate_gender_distribution(successful_results),
            "n_nodes": len(successful_results),
        }

    def _aggregate_std(
        self, results: List[Dict[str, Any]], mean_key: str, std_key: str, total_n: int
    ) -> float:
        """Aggregate standard deviation across nodes using pooled variance formula."""
        # Compute global mean
        global_mean = (
            sum(r["summary_stats"]["n_total"] * r["summary_stats"][mean_key] for r in results)
            / total_n
        )

        # Pooled variance formula: Var = sum(n_i * (var_i + (mean_i - global_mean)^2)) / N
        pooled_variance = (
            sum(
                r["summary_stats"]["n_total"]
                * (
                    r["summary_stats"][std_key] ** 2
                    + (r["summary_stats"][mean_key] - global_mean) ** 2
                )
                for r in results
            )
            / total_n
        )

        return float(np.sqrt(pooled_variance))

    def _aggregate_gender_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Aggregate gender distribution across nodes."""
        gender_counts = {}
        for r in results:
            for gender, count in r["summary_stats"]["gender_distribution"].items():
                if gender not in gender_counts:
                    gender_counts[gender] = 0
                gender_counts[gender] += count
        return gender_counts

    def _track_cohort_sizes(self, analysis_results: List[Dict[str, Any]]):
        """Track total number of records per cohort across all nodes."""
        if not hasattr(self, "cohort_n_records"):
            self.cohort_n_records = {}

        successful_results = [r for r in analysis_results if r["status"] == "success"]
        if not successful_results:
            return

        # Get cohort names from first result
        first_result = successful_results[0]

        # Initialize counts
        if "cohort_counts" in first_result:
            for cohort_name in first_result["cohort_counts"].keys():
                self.cohort_n_records[cohort_name] = 0
        else:
            for cohort_name in first_result["cohort_results"].keys():
                self.cohort_n_records[cohort_name] = 0

        # Sum cohort sizes across all nodes
        for r in successful_results:
            if "cohort_counts" in r:
                for cohort_name, count in r["cohort_counts"].items():
                    if cohort_name in self.cohort_n_records:
                        self.cohort_n_records[cohort_name] += count

    def _perform_irls_update(
        self, model_name: str, all_iterations: List[Dict[str, Any]], state: Dict[str, Any]
    ):
        """Perform one IRLS update step for a given model.

        Args:
            model_name: Name of the model
            all_iterations: List of iteration results from all nodes
            state: The model state dictionary to update

        Returns:
            True if model converged, False otherwise
        """
        if not all_iterations:
            return False

        # Get predictors from first iteration result or state
        if state["effective_predictors"] is None:
            state["effective_predictors"] = all_iterations[0]["predictors"]

        predictors = state["effective_predictors"]
        # Use 'const' to match statsmodels naming convention
        predictor_names = ["const"] + predictors

        total_info_matrix = np.sum([np.array(it["info_matrix"]) for it in all_iterations], axis=0)
        total_score_vector = np.sum([np.array(it["score_vector"]) for it in all_iterations], axis=0)
        total_deviance = np.sum([it["deviance"] for it in all_iterations])

        # Check for NaNs or Infs
        if not np.all(np.isfinite(total_info_matrix)) or not np.all(
            np.isfinite(total_score_vector)
        ):
            state["error"] = "Numerical instability: NaNs or Infs in aggregated matrices"
            return False

        # IRLS update step
        try:
            # Try standard inversion/solve first to match statsmodels behavior on near-singular matrices
            # This allows for large standard errors when separation is present, matching "Direct" results
            beta_update = np.linalg.solve(total_info_matrix, total_score_vector)
            # Use pinv for stderr calculation to match statsmodels which uses pinv for covariance
            inv_info_matrix = np.linalg.pinv(total_info_matrix)
        except np.linalg.LinAlgError:
            try:
                # Fallback to pseudo-inverse for truly singular matrices
                inv_info_matrix = np.linalg.pinv(total_info_matrix)
                beta_update = inv_info_matrix @ total_score_vector
            except np.linalg.LinAlgError:
                state["error"] = "Linear algebra error: SVD did not converge"
                return False

        # Convert current beta dict to array in correct order
        if not state["beta"]:
            current_beta = np.zeros(len(predictor_names))
        else:
            current_beta = np.array([state["beta"][pred] for pred in predictor_names])

        new_beta = current_beta + beta_update

        # Check for convergence using relative tolerance on deviance
        # This handles cases with large deviance values better
        delta_deviance = abs(total_deviance - state["deviance"])
        # Use relative tolerance similar to statsmodels
        converged = delta_deviance < (CONVERGENCE_THRESHOLD * (abs(total_deviance) + 1.0))

        # Store as dict
        state["beta"] = dict(zip(predictor_names, new_beta.tolist()))
        state["deviance"] = total_deviance
        # Always calculate stderr for current state
        # Ensure non-negative diagonal elements before sqrt
        diag_elements = np.diag(inv_info_matrix)
        # Replace negative elements (numerical noise) with 0 or small value
        diag_elements = np.maximum(diag_elements, 0)
        state["stderr"] = dict(zip(predictor_names, np.sqrt(diag_elements).tolist()))

        if converged:
            state["converged"] = True

        return converged

    def get_result(self) -> str:
        """Format the final results into a JSON string."""
        cohort_results = {}
        total_analyses = 0
        successful_analyses = 0

        for cohort_name, models in self.model_states.items():
            analyses = {}
            # Get cohort size from tracking
            n_records = 0
            if hasattr(self, "cohort_n_records") and cohort_name in self.cohort_n_records:
                n_records = self.cohort_n_records[cohort_name]

            for model_name, state in models.items():
                total_analyses += 1
                if "error" in state:
                    analyses[model_name] = {
                        "status": "failed",
                        "error": state["error"],
                        "iterations": state["iteration"],
                    }
                elif "stderr" in state and state["stderr"]:
                    # Beta and stderr are already dicts with predictor names as keys
                    analyses[model_name] = {
                        "status": "success",
                        "coefficients": state["beta"],
                        "stderr": state["stderr"],
                        "iterations": state["iteration"],
                        "converged": bool(state["converged"]) if "converged" in state else False,
                    }
                    successful_analyses += 1
                else:
                    analyses[model_name] = {
                        "status": "failed",
                        "error": "Did not converge",
                        "iterations": state["iteration"],
                    }

            cohort_results[cohort_name] = {
                "status": "completed",
                "n_records": n_records,
                "analyses": analyses,
            }

        result = {
            "overall_status": "completed",
            "summary": {
                "dataset_statistics": self.summary_stats,
                "total_cohorts": len(cohort_results),
                "total_analyses": total_analyses,
                "successful_analyses": successful_analyses,
            },
            "cohort_results": cohort_results,
        }
        return json.dumps(result, indent=2)


# Model definitions for cohorts with varying gender (include gender as predictor)
MODEL_DEFINITIONS_WITH_GENDER = {
    # name: (outcome, [predictor_1, predictor_2, ...])
    "CONDITIONTEST": ("IdiopathicHypotension", ["nt_pro_bnp_value", "age", "gender"]),
    "CONDITION_AFIB": ("AtrialFibrillation", ["nt_pro_bnp_value", "age", "gender"]),
    "CONDITION_HIS": ("HeartFailure", ["nt_pro_bnp_value", "age", "gender"]),
    "CONDITION_AFIB2": ("AtrialFibrillation", ["nt_pro_bnp_value", "age", "gender"]),
    "CONDITION_HIS2": ("HeartFailure", ["nt_pro_bnp_value", "age", "gender"]),
    "CONDITION_AFIB3": ("AtrialFibrillation", ["nt_pro_bnp_value", "age", "gender"]),
}

# Model definitions for gender-specific cohorts (exclude gender as it's constant)
MODEL_DEFINITIONS_NO_GENDER = {
    # name: (outcome, [predictor_1, predictor_2, ...])
    "CONDITIONTEST": ("IdiopathicHypotension", ["nt_pro_bnp_value", "age"]),
    "CONDITION_AFIB": ("AtrialFibrillation", ["nt_pro_bnp_value", "age"]),
    "CONDITION_HIS": ("HeartFailure", ["nt_pro_bnp_value", "age"]),
    "CONDITION_AFIB2": ("AtrialFibrillation", ["nt_pro_bnp_value", "age"]),
    "CONDITION_HIS2": ("HeartFailure", ["nt_pro_bnp_value", "age"]),
    "CONDITION_AFIB3": ("AtrialFibrillation", ["nt_pro_bnp_value", "age"]),
}

# Gender-specific cohort identifiers
GENDER_SPECIFIC_COHORTS = {
    "02_Males",
    "03_Females",
    "07_Males_Age_18_50",
    "08_Males_Age_51_80",
    "09_Males_Age_81_Inf",
    "10_Females_Age_18_50",
    "11_Females_Age_51_80",
    "12_Females_Age_81_Inf",
}


def get_model_definitions(cohort_name: str) -> Dict[str, tuple[str, List[str]]]:
    """Get appropriate model definitions for a cohort.

    Args:
        cohort_name: Name of the cohort (e.g., '01_FullCohort', '02_Males')

    Returns:
        Model definitions dict with or without gender predictor
    """
    if cohort_name in GENDER_SPECIFIC_COHORTS:
        return MODEL_DEFINITIONS_NO_GENDER
    return MODEL_DEFINITIONS_WITH_GENDER


def run_analysis_iteration(
    analyzer: Any,
    aggregator_results: Dict[str, Any] | None,
    node_id: str | None = None,
) -> Dict[str, Any]:
    """Execute a single analysis iteration for a node across all demographic cohorts.

    This is a shared method used by both local and federated analyses.

    Args:
        analyzer: The analyzer instance with prepared data
        aggregator_results: Results from the previous aggregation (or None on first iteration)
        node_id: Optional node identifier (for local analysis, comes from analyzer.node_id)

    Returns:
        Dictionary with iteration results including status, cohort_results, and summary_stats
    """
    import traceback

    if node_id is None:
        node_id = getattr(analyzer, "node_id", "unknown")

    # Handle aggregator_results safely - could be None, empty dict, or have missing keys
    iteration_num = 0
    if aggregator_results and "iteration" in aggregator_results:
        iteration_num = aggregator_results["iteration"]

    try:
        if analyzer.analysis_df is None:
            return {
                "node_id": node_id,
                "status": "error",
                "error": "Data preparation failed",
            }

        # Get demographic cohorts
        demographic_cohorts = analyzer._get_demographic_cohorts()

        # Extract model states per cohort from aggregator results
        model_states_per_cohort = {}
        if aggregator_results and "model_states" in aggregator_results:
            model_states_per_cohort = aggregator_results["model_states"]

        cohort_results = {}
        summary_stats = {}

        # Collect summary stats only on first iteration
        collect_summary = True
        if aggregator_results and "iteration" in aggregator_results:
            collect_summary = aggregator_results["iteration"] <= 1

        cohort_counts = {}

        for cohort_name, cohort_df in demographic_cohorts.items():
            cohort_counts[cohort_name] = len(cohort_df)
            if cohort_df.empty:
                cohort_results[cohort_name] = {}
                if collect_summary:
                    summary_stats[cohort_name] = {
                        "n_total": 0,
                        "nt_pro_bnp_mean": 0.0,
                        "age_mean": 0.0,
                        "atrial_fibrillation_count": 0,
                        "heart_failure_count": 0,
                    }
                continue

            # Prepare subcohorts for this demographic cohort
            analyzer._prepare_subcohorts(cohort_df)

            # Get model states for this cohort
            cohort_model_states = {}
            if cohort_name in model_states_per_cohort:
                cohort_model_states = model_states_per_cohort[cohort_name]

            iteration_results = {}
            model_defs = get_model_definitions(cohort_name)
            for name, (outcome, predictors) in model_defs.items():
                subcohort_df = None
                if name in analyzer.subcohorts:
                    subcohort_df = analyzer.subcohorts[name]

                if subcohort_df is None or subcohort_df.empty:
                    continue

                # Get current beta for this model in this cohort
                if name in cohort_model_states and "beta" in cohort_model_states[name]:
                    current_beta = cohort_model_states[name]["beta"]
                else:
                    # Initialize beta dict with zeros
                    # Use 'const' to match statsmodels naming convention
                    predictor_names = ["const"] + predictors
                    current_beta = {pred: 0.0 for pred in predictor_names}

                result = analyzer._calculate_glm_iteration(
                    subcohort_df, outcome, predictors, current_beta
                )
                if result:
                    iteration_results[name] = result

            cohort_results[cohort_name] = iteration_results

        # Collect overall dataset statistics (not per-cohort) on first iteration
        overall_stats = None
        if collect_summary:
            df = analyzer.analysis_df
            overall_stats = {
                "n_total": len(df),
                "nt_pro_bnp_mean": float(df["nt_pro_bnp_value"].mean()),
                "nt_pro_bnp_std": float(df["nt_pro_bnp_value"].std()),
                "age_mean": float(df["age"].mean()),
                "age_std": float(df["age"].std()),
                "atrial_fibrillation_count": int(df["AtrialFibrillation"].sum()),
                "heart_failure_count": int(df["HeartFailure"].sum()),
                "myocardial_infarction_count": int(df["MyocardialInfarction"].sum()),
                "stroke_count": int(df["Stroke"].sum()),
                "gender_distribution": df["gender"].value_counts().to_dict(),
            }

        return {
            "node_id": node_id,
            "status": "success",
            "cohort_results": cohort_results,
            "summary_stats": overall_stats,
            "cohort_counts": cohort_counts,
        }

    except Exception as e:
        return {
            "node_id": node_id,
            "status": "error",
            "error": f"Analysis failed: {type(e).__name__}: {str(e)}",
            "traceback": traceback.format_exc(),
        }


def run_aggregation_iteration(
    aggregator: Any,
    analysis_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Execute a single aggregation iteration across all cohorts.

    This is a shared method used by both local and federated analyses.

    Args:
        aggregator: The aggregator instance with state and methods
        analysis_results: Results from all analyzers in this iteration

    Returns:
        Dictionary with aggregation results including model_states and summary
    """
    aggregator.iteration += 1
    print(f"\n[Iteration {aggregator.iteration}]")

    successful_nodes = [r for r in analysis_results if r["status"] == "success"]
    if not successful_nodes:
        return {"error": "No successful nodes in this iteration."}

    if aggregator.iteration == 1:
        aggregator._initialize_model_states(analysis_results)
        aggregator._aggregate_summary_stats(analysis_results)
        # Track total records per cohort across all nodes
        aggregator._track_cohort_sizes(analysis_results)

    # Aggregate score vectors and info matrices for each cohort and model
    for cohort_name, models in aggregator.model_states.items():
        cohort_label = cohort_name.split("_", 1)[1] if "_" in cohort_name else cohort_name

        for model_name, state in models.items():
            if state["converged"]:
                continue

            # Collect results from all nodes for this cohort and model
            all_iterations = [
                r["cohort_results"][cohort_name][model_name]
                for r in successful_nodes
                if "cohort_results" in r
                and cohort_name in r["cohort_results"]
                and model_name in r["cohort_results"][cohort_name]
            ]

            if not all_iterations:
                continue

            converged = aggregator._perform_irls_update(model_name, all_iterations, state)
            state["iteration"] = aggregator.iteration
            if converged:
                print(f"  {cohort_label}/{model_name}: CONVERGED")

    # Check if all models have converged
    all_converged = all(
        state["converged"] if "converged" in state else False
        for models in aggregator.model_states.values()
        for state in models.values()
    )

    if all_converged:
        # Parse the JSON string result back to dict for consistency
        result_json = aggregator.get_result()
        return json.loads(result_json)

    # Continue iterations
    return {
        "iteration": aggregator.iteration,
        "model_states": aggregator.model_states,
        "summary": aggregator.summary_stats,
    }
