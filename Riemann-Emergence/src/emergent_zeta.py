"""
Construction of emergent zeta-like function ζ_TST(s)
"""
def emergent_zeta(s, eigenvalues):
    product = 1.0
    for λ in eigenvalues:
        product *= (1 - s / λ)
    return product
