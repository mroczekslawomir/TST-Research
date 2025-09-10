"""
Construct field functionals used in TST simulations
"""

import numpy as np

def calculate_gradient_term(fields):
    return sum(np.sum(g**2) for g in np.gradient(fields))

def calculate_potential_term(fields, lambda_val=0.1, v_ev=1.0):
    return lambda_val * np.sum((fields**2 - v_ev**2)**2)

def calculate_topological_charge(fields):
    # Approximate Hopf invariant for scalar field
    return np.sum(np.sin(fields) * np.cos(fields))

def full_functional(fields, lambda_val=0.1, v_ev=1.0):
    grad = calculate_gradient_term(fields)
    pot = calculate_potential_term(fields, lambda_val, v_ev)
    topo = calculate_topological_charge(fields)
    return grad + pot + topo
