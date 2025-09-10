"""
Test emergent gauge symmetries and anomaly cancellation
"""

def test_gauge_charges(simulation):
    for group in ['SU(3)', 'SU(2)', 'U(1)']:
        charge = simulation[f'{group}_charge']
        assert abs(charge - round(charge)) < 0.01, f"{group} charge not conserved"

def test_anomaly(simulation):
    anomaly = simulation['fermionic_anomaly']
    assert abs(anomaly) < 1e-3, "Anomaly too large"
