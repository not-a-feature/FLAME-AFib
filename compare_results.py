"""Compare results from local and direct GLM analysis.

This script loads the results from both analysis methods and compares them to verify
that the federated analysis produces mathematically equivalent results to direct GLM
fitting on pooled data.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

__author__ = "Jules Kreuer, jules.kreuer@uni-tuebingen.de"
__version__ = "0.1.0"

# Tolerance for floating point comparison
RELATIVE_TOLERANCE = 1e-4  # 0.01%
ABSOLUTE_TOLERANCE = 1e-6


def load_json_file(filepath: str) -> Dict[str, Any]:
    """Load JSON file and return as dictionary.

    Handles common JSON formatting issues:
    - Trailing commas in objects and arrays
    - Python-style booleans (True/False -> true/false)
    - Python-style None -> null
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace Python-style values with JSON equivalents
    # Handle booleans (with and without trailing commas/whitespace)
    content = re.sub(r"'", '"', content)
    content = re.sub(r"\bTrue\b", "true", content)
    content = re.sub(r"\bFalse\b", "false", content)
    content = re.sub(r"\bNone\b", "null", content)

    # Remove trailing commas before closing braces/brackets
    # Pattern: comma followed by optional whitespace and then } or ]
    content = re.sub(r",(\s*[}\]])", r"\1", content)

    return json.loads(content)


def compare_floats(val1: float, val2: float, path: str) -> Tuple[bool, str]:
    """Compare two float values with tolerance."""
    if not isinstance(val1, (int, float)) or not isinstance(val2, (int, float)):
        return False, f"Type mismatch at {path}: {type(val1).__name__} vs {type(val2).__name__}"

    val1_f = float(val1)
    val2_f = float(val2)

    # Check if both are NaN or Inf
    if np.isnan(val1_f) and np.isnan(val2_f):
        return True, ""
    if np.isinf(val1_f) and np.isinf(val2_f):
        if np.sign(val1_f) == np.sign(val2_f):
            return True, ""
        return False, f"Infinity sign mismatch at {path}: {val1_f} vs {val2_f}"

    # Check for NaN/Inf mismatch
    if np.isnan(val1_f) or np.isnan(val2_f) or np.isinf(val1_f) or np.isinf(val2_f):
        return False, f"NaN/Inf mismatch at {path}: {val1_f} vs {val2_f}"

    # Use numpy's isclose for robust comparison
    if np.isclose(val1_f, val2_f, rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE):
        return True, ""

    diff = abs(val1_f - val2_f)
    rel_diff = diff / max(abs(val1_f), abs(val2_f)) if max(abs(val1_f), abs(val2_f)) > 0 else 0
    return (
        False,
        f"Value mismatch at {path}: {val1_f} vs {val2_f} (diff: {diff:.2e}, rel: {rel_diff:.2e})",
    )


def compare_values(val1: Any, val2: Any, path: str = "root") -> Tuple[bool, List[str]]:
    """Recursively compare two values, handling floats with tolerance."""
    errors = []

    # Type check
    if type(val1) != type(val2):
        errors.append(f"Type mismatch at {path}: {type(val1).__name__} vs {type(val2).__name__}")
        return False, errors

    # Handle None
    if val1 is None:
        return True, errors

    # Handle bool (before numeric check since bool is subclass of int)
    if isinstance(val1, bool):
        if val1 != val2:
            errors.append(f"Boolean mismatch at {path}: {val1} vs {val2}")
        return len(errors) == 0, errors

    # Handle numeric values (int, float)
    if isinstance(val1, (int, float)):
        match, error = compare_floats(val1, val2, path)
        if not match:
            errors.append(error)
        return len(errors) == 0, errors

    # Handle strings
    if isinstance(val1, str):
        if val1 != val2:
            errors.append(f"String mismatch at {path}: '{val1}' vs '{val2}'")
        return len(errors) == 0, errors

    # Handle dictionaries
    if isinstance(val1, dict):
        keys1 = set(val1.keys())
        keys2 = set(val2.keys())

        if keys1 != keys2:
            missing_in_2 = keys1 - keys2
            missing_in_1 = keys2 - keys1
            if missing_in_2:
                errors.append(f"Keys missing in second dict at {path}: {missing_in_2}")
            if missing_in_1:
                errors.append(f"Keys missing in first dict at {path}: {missing_in_1}")
            return False, errors

        for key in keys1:
            match, sub_errors = compare_values(val1[key], val2[key], f"{path}.{key}")
            errors.extend(sub_errors)

        return len(errors) == 0, errors

    # Handle lists
    if isinstance(val1, list):
        if len(val1) != len(val2):
            errors.append(f"List length mismatch at {path}: {len(val1)} vs {len(val2)}")
            return False, errors

        for i, (item1, item2) in enumerate(zip(val1, val2)):
            match, sub_errors = compare_values(item1, item2, f"{path}[{i}]")
            errors.extend(sub_errors)

        return len(errors) == 0, errors

    # Fallback to direct comparison
    if val1 != val2 and not path.endswith("iterations"):
        errors.append(f"Value mismatch at {path}: {val1} vs {val2}")

    return len(errors) == 0, errors


def print_summary_comparison(local_data: Dict[str, Any], direct_data: Dict[str, Any]):
    """Print a summary comparison of key metrics."""

    local_cohorts = local_data.get("cohort_results", {})
    direct_cohorts = direct_data.get("cohort_results", {})

    for cohort_name in sorted(local_cohorts.keys()):
        if cohort_name not in direct_cohorts:
            print(f"\n[{cohort_name}] MISSING in direct analysis")
            continue

        local_cohort = local_cohorts[cohort_name]
        direct_cohort = direct_cohorts[cohort_name]

        print(f"\n[{cohort_name}]")
        print(f"  Status: Local={local_cohort.get('status')}, Direct={direct_cohort.get('status')}")
        print(
            f"  Records: Local={local_cohort.get('n_records')}, Direct={direct_cohort.get('n_records')}"
        )

        if local_cohort.get("status") != "completed" or direct_cohort.get("status") != "completed":
            continue

        local_analyses = local_cohort.get("analyses", {})
        direct_analyses = direct_cohort.get("analyses", {})

        for analysis_name in sorted(local_analyses.keys()):
            if analysis_name not in direct_analyses:
                print(f"  [{analysis_name}] MISSING in direct analysis")
                continue

            local_model = local_analyses[analysis_name]
            direct_model = direct_analyses[analysis_name]

            if local_model.get("status") != "success" or direct_model.get("status") != "success":
                print(
                    f"  [{analysis_name}] Status: Local={local_model.get('status')}, Direct={direct_model.get('status')}"
                )
                continue

            print(f"  [{analysis_name}]")
            print(
                f"    Converged: Local={local_model.get('converged')}, Direct={direct_model.get('converged')}"
            )
            print(
                f"    Iterations: Local={local_model.get('iterations')}, Direct={direct_model.get('iterations')}"
            )

            local_coef = local_model.get("coefficients", {})
            direct_coef = direct_model.get("coefficients", {})

            print(f"    Coefficients:")
            for coef_name in sorted(local_coef.keys()):
                if coef_name in direct_coef:
                    diff = abs(local_coef[coef_name] - direct_coef[coef_name])
                    match = "✓" if diff < 1e-4 else "✗"
                    print(
                        f"      {match} {coef_name:20s}: Local={local_coef[coef_name]:12.6f}, Direct={direct_coef[coef_name]:12.6f}, Diff={diff:.2e}"
                    )


def main():
    """Load and compare results from local and direct analysis."""
    parser = argparse.ArgumentParser(description="Compare two result JSON files (local vs direct).")
    parser.add_argument(
        "local",
        nargs="?",
        default="output/local.json",
        help="Path to local (federated) results JSON (default: %(default)s)",
    )
    parser.add_argument(
        "direct",
        nargs="?",
        default="output/direct.json",
        help="Path to direct GLM results JSON (default: %(default)s)",
    )

    args = parser.parse_args()
    local_path = args.local
    direct_path = args.direct

    print(f"Local:  {local_path}")
    print(f"Direct: {direct_path}")

    try:
        local_data = load_json_file(local_path)
        direct_data = load_json_file(direct_path)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nPlease generate the output files before comparing.")
        sys.exit(2)

    print(f"\n[INFO] Files loaded successfully")

    print_summary_comparison(local_data, direct_data)

    print("\n\n" + "=" * 60)
    print("DETAILED COMPARISON")

    match, errors = compare_values(local_data, direct_data)

    if match:
        print("\n✓ SUCCESS: Results match within tolerance!")
        print(f"  Relative tolerance: {RELATIVE_TOLERANCE}")
        print(f"  Absolute tolerance: {ABSOLUTE_TOLERANCE}")
        sys.exit(0)
    else:
        print(f"\n✗ FAILURE: Found {len(errors)} differences:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
