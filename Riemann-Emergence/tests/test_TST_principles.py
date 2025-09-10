"""
Test fundamental principles of Temporal Synchronization Theory (TST)
"""

def test_minimal_desynchronization(simulation):
    initial_I = simulation['initial_desynchronization']
    final_I = simulation['final_desynchronization']
    assert final_I < initial_I, "Desynchronization did not decrease"

def test_non_zero_proper_time(simulation):
    proper_time = simulation['proper_time_operator']
    assert all(t > 0 for t in proper_time), "Proper time violated (t ≤ 0)"
