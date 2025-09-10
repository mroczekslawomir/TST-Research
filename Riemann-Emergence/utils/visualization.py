"""
Visualization utilities for TST simulations
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_eigenvalue_distribution(eigenvalues, title="Eigenvalue Spectrum", filename=None):
    plt.figure(figsize=(10, 6))
    plt.hist(eigenvalues, bins=50, color='skyblue', edgecolor='black')
    plt.axvline(0.5, color='red', linestyle='--', label='Re(s) = 0.5')
    plt.title(title)
    plt.xlabel("Eigenvalue")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
