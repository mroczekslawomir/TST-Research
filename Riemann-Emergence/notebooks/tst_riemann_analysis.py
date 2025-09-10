# TST Riemann Hypothesis Analysis Notebook

# 📦 Importy
import numpy as np
import networkx as nx
from scipy.sparse import diags, identity
from scipy.sparse.linalg import eigsh
from scipy.special import erf
import matplotlib.pyplot as plt
# 🧠 Parametry sieci
N = 1000
k = 4
p = 0.1

# 🔧 Generowanie sieci Watts–Strogatz
G = nx.watts_strogatz_graph(N, k, p, seed=42)

# 🧮 Konstrukcja operatora desynchronizacji Î
L = nx.laplacian_matrix(G).astype(float)
degrees = np.array([d for _, d in G.degree()])
I_sync = identity(N, format='csr') * 1.0
I_ent = diags(0.1 * degrees, 0, format='csr')
I_operator = 1.0 * L + I_sync + I_ent

# 📈 Obliczenie widma operatora fluktuacji
eigenvalues, _ = eigsh(I_operator, k=100, which='SM')
norm = (eigenvalues - np.min(eigenvalues)) / (np.max(eigenvalues) - np.min(eigenvalues))

# 📊 Wizualizacja histogramu
plt.figure(figsize=(10, 6))
plt.hist(norm, bins=50, color='skyblue', edgecolor='black')
plt.axvline(0.5, color='red', linestyle='--', label='Re(s) = 0.5')
plt.title("Normalized Eigenvalue Spectrum")
plt.xlabel("Normalized Eigenvalue")
plt.ylabel("Count")
plt.legend()
plt.grid(True)
plt.show()

# 📐 Obliczenie p-value dla hipotezy Riemanna
in_band = np.sum((norm >= 0.45) & (norm <= 0.55))
expected = 100 * 0.1
z = (in_band - expected) / np.sqrt(expected)
p_value = 2 * (1 - 0.5 * (1 + erf(z / np.sqrt(2)))) if in_band > expected else 1.0

# 📢 Wyniki
print(f"Eigenvalues in [0.45, 0.55]: {in_band}/100")
print(f"p-value: {p_value:.2e}")
