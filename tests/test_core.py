"""
Test suite for MGDStars core functionality.

This module tests the core MGDStars class including:
- Initialization and configuration
- Symbolic expression derivation
- Numerical solving
- Physical property evaluation
- Model-specific behavior
"""

import pytest
import numpy as np
from mgdstars import MGDStars


class TestMGDStarsInitialization:
    """Test MGDStars class initialization and basic setup."""
    
    def test_default_initialization(self):
        """Test default initialization parameters."""
        star = MGDStars()
        assert star.alpha == 0.1
        assert star.k2_factor == 8 * np.pi
        assert star.seed_model == 'TolmanIV'
        assert not star.solution_found
    
    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        star = MGDStars(alpha=0.05, seed_model='HeintzmannIIa')
        assert star.alpha == 0.05
        assert star.seed_model == 'HeintzmannIIa'
    
    def test_invalid_seed_model(self):
        """Test handling of invalid seed model."""
        with pytest.raises(ValueError):
            star = MGDStars(seed_model='InvalidModel')


class TestSeedModels:
    """Test different seed model implementations."""
    
    @pytest.mark.parametrize("seed_model", ['TolmanIV', 'HeintzmannIIa', 'WymanIIa'])
    def test_seed_model_creation(self, seed_model):
        """Test that all seed models can be created successfully."""
        star = MGDStars(seed_model=seed_model)
        assert star.seed_model == seed_model
        
    def test_tolmaniv_convergence(self):
        """Test TolmanIV model convergence for standard case."""
        star = MGDStars(alpha=0.1, seed_model='TolmanIV')
        result = star.solve_star_configuration(M_val=1.4, R_val=12.0)
        
        assert result['success'], f"TolmanIV failed to converge: {result.get('error', 'Unknown')}"
        assert result['residual'] < 1e-6
        assert star.solution_found
    
    def test_heintzmanniia_convergence(self):
        """Test HeintzmannIIa model convergence with appropriate alpha."""
        star = MGDStars(alpha=0.01, seed_model='HeintzmannIIa')
        result = star.solve_star_configuration(M_val=1.4, R_val=12.0)
        
        assert result['success'], f"HeintzmannIIa failed to converge: {result.get('error', 'Unknown')}"
        assert result['residual'] < 1e-6
    
    def test_wymaniia_convergence(self):
        """Test WymanIIa model convergence with appropriate alpha."""
        star = MGDStars(alpha=0.005, seed_model='WymanIIa')
        result = star.solve_star_configuration(M_val=1.4, R_val=12.0)
        
        assert result['success'], f"WymanIIa failed to converge: {result.get('error', 'Unknown')}"
        assert result['residual'] < 1e-6


class TestPhysicalProperties:
    """Test physical property calculations."""
    
    def test_evaluate_at_radius(self):
        """Test physical property evaluation at different radii."""
        star = MGDStars(alpha=0.1, seed_model='TolmanIV')
        result = star.solve_star_configuration(M_val=1.4, R_val=12.0)
        
        assert result['success']
        
        # Test single radius
        r_vals = np.array([6.0])
        properties = star.evaluate_at_radius(r_vals)
        
        assert 'rho' in properties
        assert 'pr' in properties
        assert 'pt' in properties
        assert 'mu' in properties
        assert 'e_nu' in properties
        
        # Check that all values are finite and positive where expected
        assert np.all(np.isfinite(properties['rho']))
        assert np.all(properties['rho'] >= 0)
        
    def test_evaluate_at_multiple_radii(self):
        """Test evaluation at multiple radii."""
        star = MGDStars(alpha=0.1, seed_model='TolmanIV')
        star.solve_star_configuration(M_val=1.4, R_val=12.0)
        
        r_vals = np.linspace(0.1, 11.9, 10)
        properties = star.evaluate_at_radius(r_vals)
        
        # Check array dimensions
        assert len(properties['rho']) == len(r_vals)
        assert len(properties['pr']) == len(r_vals)
        
        # Check monotonicity (density should decrease outward)
        assert properties['rho'][0] > properties['rho'][-1]
    
    def test_no_solution_error(self):
        """Test that evaluation fails when no solution exists."""
        star = MGDStars(alpha=0.1, seed_model='TolmanIV')
        
        with pytest.raises(ValueError, match="No solution found"):
            star.evaluate_at_radius([6.0])


class TestNumericalStability:
    """Test numerical stability and edge cases."""
    
    def test_extreme_compactness(self):
        """Test behavior with extreme compactness values."""
        star = MGDStars(alpha=0.005, seed_model='TolmanIV')
        
        # Very high compactness (should be challenging)
        result = star.solve_star_configuration(M_val=2.5, R_val=10.0)  # β ≈ 0.25
        
        # May fail, but should not crash
        assert 'success' in result
        assert 'residual' in result
    
    def test_small_alpha_values(self):
        """Test convergence with very small alpha values."""
        star = MGDStars(alpha=0.001, seed_model='TolmanIV')
        result = star.solve_star_configuration(M_val=1.4, R_val=12.0)
        
        # Should converge with small alpha
        assert result['success']
    
    def test_large_alpha_values(self):
        """Test behavior with large alpha values."""
        star = MGDStars(alpha=0.5, seed_model='TolmanIV')
        result = star.solve_star_configuration(M_val=1.4, R_val=12.0)
        
        # May fail, but should handle gracefully
        assert 'success' in result
    
    @pytest.mark.parametrize("tolerance", [1e-6, 1e-8, 1e-10])
    def test_convergence_tolerances(self, tolerance):
        """Test different convergence tolerances."""
        star = MGDStars(alpha=0.1, seed_model='TolmanIV')
        result = star.solve_star_configuration(
            M_val=1.4, R_val=12.0, tolerance=tolerance
        )
        
        if result['success']:
            assert result['residual'] <= tolerance * 10  # Allow some margin


class TestErrorHandling:
    """Test error handling and robustness."""
    
    def test_invalid_mass_radius(self):
        """Test handling of invalid mass/radius values."""
        star = MGDStars(alpha=0.1, seed_model='TolmanIV')
        
        # Negative values
        result = star.solve_star_configuration(M_val=-1.0, R_val=12.0)
        assert not result['success']
        
        result = star.solve_star_configuration(M_val=1.4, R_val=-12.0)
        assert not result['success']
        
        # Zero values
        result = star.solve_star_configuration(M_val=0.0, R_val=12.0)
        assert not result['success']
    
    def test_convergence_failure_diagnosis(self):
        """Test convergence failure diagnosis."""
        star = MGDStars(alpha=1.0, seed_model='TolmanIV')  # Likely to fail
        
        # This should not raise an exception
        diagnosis = star.diagnose_convergence_failure(M_val=1.4, R_val=12.0)
        
        assert isinstance(diagnosis, dict)
        assert 'strategies_tested' in diagnosis


if __name__ == '__main__':
    pytest.main([__file__])
