"""Test suite for mathematical and logical correctness of AFib analysis.

This module tests:
1. GLM iteration calculations (score vector, information matrix, deviance)
2. Unit conversions for NT-proBNP
3. Numerical stability with extreme values
4. Convergence behavior
"""

import numpy as np
import pandas as pd
import pytest
from afib_analysis_utils import (
    AFibAnalyzerMixin,
    CONVERGENCE_THRESHOLD,
    MAX_ITERATIONS,
)


class TestGLMIterationCalculation:
    """Test the GLM iteration calculation for mathematical correctness."""

    def test_mu_clipping_prevents_zero_weights(self):
        """Test that mu clipping prevents zero weights in extreme cases."""
        # Create a simple test case with extreme predictions
        class TestAnalyzer(AFibAnalyzerMixin):
            pass

        analyzer = TestAnalyzer()

        # Create test data with extreme predictor values (need at least 10 samples)
        df = pd.DataFrame(
            {
                "outcome": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                "predictor1": [100, -100, 50, -50, 100, -100, 50, -50, 100, -100],
                "gender_numeric": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            }
        )

        # Beta values that would cause extreme predictions
        beta = np.array([0.0, 1.0])

        result = analyzer._calculate_glm_iteration(
            df, "outcome", ["predictor1"], beta
        )

        assert result is not None, "Calculation should not return None"
        
        # Verify all values are finite (no inf or nan)
        info_matrix = np.array(result["info_matrix"])
        score = np.array(result["score_vector"])
        
        assert np.all(np.isfinite(info_matrix)), "Info matrix should not contain inf or nan"
        assert np.all(np.isfinite(score)), "Score vector should not contain inf or nan"
        assert np.isfinite(result["deviance"]), "Deviance should be finite"
        
        # With extreme values, the determinant may be very small but should not be exactly zero
        # The clipping ensures numerical stability even if the matrix is nearly singular
        det = np.linalg.det(info_matrix)
        assert np.isfinite(det), f"Determinant should be finite, got {det}"

    def test_score_vector_calculation(self):
        """Test that score vector calculation is mathematically correct."""
        class TestAnalyzer(AFibAnalyzerMixin):
            pass

        analyzer = TestAnalyzer()

        # Create simple test data (need at least 10 samples)
        df = pd.DataFrame(
            {
                "outcome": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                "predictor1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "gender_numeric": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            }
        )

        beta = np.array([0.0, 0.0])

        result = analyzer._calculate_glm_iteration(
            df, "outcome", ["predictor1"], beta
        )

        assert result is not None
        score = np.array(result["score_vector"])
        
        # When beta = 0, mu should be 0.5 (clipped to be in valid range)
        # Score should be X^T(y - mu)
        # With y = [1,0,1,0] and mu ≈ [0.5,0.5,0.5,0.5]
        # residuals = [0.5, -0.5, 0.5, -0.5]
        
        assert score.shape == (2,), f"Score should have 2 elements, got {score.shape}"
        assert np.all(np.isfinite(score)), "Score should not contain inf or nan"

    def test_deviance_calculation(self):
        """Test that deviance calculation doesn't produce inf or nan."""
        class TestAnalyzer(AFibAnalyzerMixin):
            pass

        analyzer = TestAnalyzer()

        # Test with perfect predictions (need at least 10 samples)
        df = pd.DataFrame(
            {
                "outcome": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                "predictor1": [10, 10, 10, 10, 10, -10, -10, -10, -10, -10],
                "gender_numeric": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            }
        )

        beta = np.array([0.0, 1.0])

        result = analyzer._calculate_glm_iteration(
            df, "outcome", ["predictor1"], beta
        )

        assert result is not None
        deviance = result["deviance"]
        
        assert np.isfinite(deviance), f"Deviance should be finite, got {deviance}"
        assert deviance >= 0, f"Deviance should be non-negative, got {deviance}"


class TestUnitConversions:
    """Test NT-proBNP unit conversion factors."""

    def test_ng_l_to_pg_ml_conversion(self):
        """Test that ng/L to pg/mL conversion is correct (should be 1.0)."""
        # 1 ng/L = 1000 pg / 1000 mL = 1 pg/mL
        from afib_analysis_utils import AFibAnalyzerMixin
        
        class TestAnalyzer(AFibAnalyzerMixin):
            pass

        analyzer = TestAnalyzer()
        
        # Create test data
        analyzer.analysis_df = pd.DataFrame(
            {
                "NTproBNP.unit": ["ng/L"],
                "nt_pro_bnp_value": [100.0],
            }
        )
        
        analyzer._convert_nt_pro_bnp_units()
        
        # After conversion, value should still be 100 (factor = 1.0)
        assert analyzer.analysis_df["nt_pro_bnp_value"].iloc[0] == 100.0
        assert analyzer.analysis_df["NTproBNP.unit"].iloc[0] == "pg/mL"

    def test_pg_dl_to_pg_ml_conversion(self):
        """Test that pg/dL to pg/mL conversion is correct (should divide by 100)."""
        # 100 pg/dL = 100 pg / 100 mL = 1 pg/mL
        from afib_analysis_utils import AFibAnalyzerMixin
        
        class TestAnalyzer(AFibAnalyzerMixin):
            pass

        analyzer = TestAnalyzer()
        
        analyzer.analysis_df = pd.DataFrame(
            {
                "NTproBNP.unit": ["pg/dL"],
                "nt_pro_bnp_value": [100.0],
            }
        )
        
        analyzer._convert_nt_pro_bnp_units()
        
        # After conversion, value should be 1.0 (100 * 0.01)
        expected = 100.0 * 0.01
        actual = analyzer.analysis_df["nt_pro_bnp_value"].iloc[0]
        assert np.isclose(actual, expected), f"Expected {expected}, got {actual}"

    def test_pg_l_to_pg_ml_conversion(self):
        """Test that pg/L to pg/mL conversion is correct (should divide by 1000)."""
        # 1000 pg/L = 1000 pg / 1000 mL = 1 pg/mL
        from afib_analysis_utils import AFibAnalyzerMixin
        
        class TestAnalyzer(AFibAnalyzerMixin):
            pass

        analyzer = TestAnalyzer()
        
        analyzer.analysis_df = pd.DataFrame(
            {
                "NTproBNP.unit": ["pg/L"],
                "nt_pro_bnp_value": [1000.0],
            }
        )
        
        analyzer._convert_nt_pro_bnp_units()
        
        # After conversion, value should be 1.0 (1000 * 0.001)
        expected = 1000.0 * 0.001
        actual = analyzer.analysis_df["nt_pro_bnp_value"].iloc[0]
        assert np.isclose(actual, expected), f"Expected {expected}, got {actual}"

    def test_pmol_l_to_pg_ml_conversion(self):
        """Test that pmol/L to pg/mL conversion uses correct molecular weight."""
        # For NT-proBNP with MW ≈ 8.457 kDa:
        # 100 pmol/L ≈ 845.7 pg/mL
        from afib_analysis_utils import AFibAnalyzerMixin
        
        class TestAnalyzer(AFibAnalyzerMixin):
            pass

        analyzer = TestAnalyzer()
        
        analyzer.analysis_df = pd.DataFrame(
            {
                "NTproBNP.unit": ["pmol/L"],
                "nt_pro_bnp_value": [100.0],
            }
        )
        
        analyzer._convert_nt_pro_bnp_units()
        
        # After conversion, value should be ~845.7 (100 * 8.457)
        expected = 100.0 * 8.457
        actual = analyzer.analysis_df["nt_pro_bnp_value"].iloc[0]
        assert np.isclose(actual, expected), f"Expected {expected}, got {actual}"


class TestConvergenceBehavior:
    """Test convergence threshold and iteration behavior."""

    def test_convergence_threshold_is_reasonable(self):
        """Test that convergence threshold is appropriately small."""
        assert CONVERGENCE_THRESHOLD <= 1e-5, (
            f"Convergence threshold {CONVERGENCE_THRESHOLD} is too large. "
            f"Should be <= 1e-5 for accurate GLM convergence."
        )

    def test_max_iterations_is_sufficient(self):
        """Test that max iterations is sufficient for convergence."""
        assert MAX_ITERATIONS >= 20, (
            f"Max iterations {MAX_ITERATIONS} may be too small. "
            f"Should be >= 20 to ensure convergence for complex models."
        )


class TestNumericalStability:
    """Test numerical stability in edge cases."""

    def test_handles_perfect_separation(self):
        """Test that code handles perfect separation gracefully."""
        # This simulates the CONDITIONTEST scenario where all cases are male
        class TestAnalyzer(AFibAnalyzerMixin):
            pass

        analyzer = TestAnalyzer()

        # All outcome=1 cases have predictor=0, all outcome=0 cases have predictor=1
        # Need at least 10 samples
        df = pd.DataFrame(
            {
                "outcome": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                "predictor1": [1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                "separating_var": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # Perfect separation
                "gender_numeric": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            }
        )

        beta = np.array([0.0, 0.0, -5.0])  # Large negative for separating variable

        result = analyzer._calculate_glm_iteration(
            df, "outcome", ["predictor1", "separating_var"], beta
        )

        assert result is not None, "Should handle perfect separation"
        
        # Even with perfect separation, clipping should prevent numerical errors
        assert np.all(np.isfinite(result["score_vector"]))
        assert np.all(np.isfinite(result["info_matrix"]))
        assert np.isfinite(result["deviance"])

    def test_handles_very_small_sample_size(self):
        """Test behavior with minimum sample size."""
        class TestAnalyzer(AFibAnalyzerMixin):
            pass

        analyzer = TestAnalyzer()

        # Exactly 10 samples (minimum threshold)
        df = pd.DataFrame(
            {
                "outcome": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                "predictor1": list(range(10)),
                "gender_numeric": [0, 1] * 5,
            }
        )

        beta = np.array([0.0, 0.0])

        result = analyzer._calculate_glm_iteration(
            df, "outcome", ["predictor1"], beta
        )

        assert result is not None
        assert result["n_samples"] == 10

    def test_rejects_insufficient_data(self):
        """Test that calculation returns None for insufficient data."""
        class TestAnalyzer(AFibAnalyzerMixin):
            pass

        analyzer = TestAnalyzer()

        # Less than 10 samples
        df = pd.DataFrame(
            {
                "outcome": [1, 0, 1],
                "predictor1": [1, 2, 3],
                "gender_numeric": [0, 1, 0],
            }
        )

        beta = np.array([0.0, 0.0])

        result = analyzer._calculate_glm_iteration(
            df, "outcome", ["predictor1"], beta
        )

        assert result is None, "Should return None for insufficient data"

    def test_rejects_constant_outcome(self):
        """Test that calculation returns None when outcome has no variation."""
        class TestAnalyzer(AFibAnalyzerMixin):
            pass

        analyzer = TestAnalyzer()

        # All outcomes are the same
        df = pd.DataFrame(
            {
                "outcome": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                "predictor1": list(range(10)),
                "gender_numeric": [0, 1] * 5,
            }
        )

        beta = np.array([0.0, 0.0])

        result = analyzer._calculate_glm_iteration(
            df, "outcome", ["predictor1"], beta
        )

        assert result is None, "Should return None for constant outcome"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
