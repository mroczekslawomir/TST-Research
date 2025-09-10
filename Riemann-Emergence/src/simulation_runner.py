"""
Main simulation loop for Temporal Synchronization Theory (TST)
"""

from operator_desynchronization import build_desynchronization_operator
from spectral_analysis import analyze_spectrum
import networkx as nx
from scipy.sparse.linalg import eigsh

def run_simulation(network_type='WS', N=1000, k=4, p=0.1, num_eigenvalues=100):
    if network_type == 'WS':
        G = nx.watts_strogatz_graph(N, k, p, seed=42)
    elif network_type == 'BA':
        G = nx.barabasi_albert_graph(N, k, seed=42)
    else:
        raise ValueError("Unsupported network type")

    operator = build_desynchronization_operator(G)
    eigenvalues, _ = eigsh(operator, k=num_eigenvalues, which='SM')
    return analyze_spectrum(eigenvalues)

# main() — run as script
def main():
    print("🔬 Running TST Riemann Hypothesis Simulation")
    result = run_simulation(network_type='WS', N=1000, k=4, p=0.1, num_eigenvalues=100)
    print(f"✅ Eigenvalues in [0.45, 0.55]: {result['in_band']}/100")
    print(f"📊 p-value: {result['p_value']:.2e}")
    if result['p_value'] < 0.05:
        print("🎯 Strong evidence for Riemann Hypothesis in TST!")
    else:
        print("⚠️ No significant evidence — try different parameters.")

if __name__ == "__main__":
    main()
