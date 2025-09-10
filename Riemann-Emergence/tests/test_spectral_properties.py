"""
Test spectral properties of the fluctuation operator δ²⟨Î⟩/δΨ²
"""

import numpy as np
from spectral_analysis import analyze_spectrum

def test_eigenvalue_band():
    # Simulated eigenvalues with clustering around 0.5
    eigenvalues = np.array([0.48, 0.49, 0.50, 0.51, 0.52] + list(np.random.rand(95)))
    result = analyze_spectrum(eigenvalues)
    assert result['in_band'] >= 5, "Too few eigenvalues in critical band [0.45, 0.55]"
    assert result['p_value'] < 0.1, "Spectrum not statistically significant"
