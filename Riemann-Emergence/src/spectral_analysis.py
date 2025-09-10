"""
Spectral analysis of fluctuation operator δ²⟨Î⟩/δΨ²
"""
import numpy as np
from scipy.special import erf

def analyze_spectrum(eigenvalues):
    norm = (eigenvalues - np.min(eigenvalues)) / (np.max(eigenvalues) - np.min(eigenvalues))
    in_band = np.sum((norm >= 0.45) & (norm <= 0.55))
    very_close = np.sum(np.abs(norm - 0.5) < 0.01)
    expected = len(norm) * 0.1
    z = (in_band - expected) / np.sqrt(expected)
    p_value = 2 * (1 - 0.5 * (1 + erf(z / np.sqrt(2)))) if in_band > expected else 1.0
    return {
        'normalized_eigenvalues': norm,
        'in_band': in_band,
        'very_close': very_close,
        'p_value': p_value,
        'mean_deviation': np.mean(np.abs(norm - 0.5)),
        'std_deviation': np.std(norm)
    }
