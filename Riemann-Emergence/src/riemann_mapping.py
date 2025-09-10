"""
Mapping spectrum to zeros of the Riemann zeta function
"""
def map_to_zeta_zeros(eigenvalues):
    return [0.5 + 1j * (ev - 0.5) * 100 for ev in eigenvalues]
