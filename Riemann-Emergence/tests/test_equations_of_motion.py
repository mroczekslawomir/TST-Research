"""
Test correctness of equations of motion in TST
"""

def test_schrodinger_equation(simulation):
    deviation = simulation['schrodinger_deviation']
    assert deviation < 1e-4, "Schrödinger-like equation not satisfied"

def test_einstein_equations(simulation):
    deviation = simulation['einstein_deviation']
    assert deviation < 1e-3, "Einstein field equations not satisfied"
