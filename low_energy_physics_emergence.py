import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Parametry sieci
N = 128
m = 3
Λ = 647  # MeV

# 1. Generowanie sieci Barabási–Albert
G = nx.barabasi_albert_graph(N, m)

# 2. Przypisanie faz do węzłów (próżnia: pełna synchronizacja)
phases_vacuum = np.zeros(N)

# 3. Konfiguracja cząstki tau: lokalna dyssynchronizacja
phases_tau = np.zeros(N)
center = N // 2
phases_tau[center] = np.pi / 2  # lokalny defekt
for neighbor in G.neighbors(center):
    phases_tau[neighbor] = np.pi / 4

# 4. Definicja operatora dyssynchronizacji ⟨Î⟩
def I_sync(phases):
    return sum((phases[i] - phases[j])**2 for i, j in G.edges)

def I_grad(phases):
    return sum(abs(phases[i] - phases[j]) for i, j in G.edges)

def I_ent(phases):
    hist, _ = np.histogram(phases, bins=10, range=(-np.pi, np.pi), density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log(hist))

def I_total(phases):
    return I_sync(phases) + I_grad(phases) + I_ent(phases)

# 5. Obliczenie wartości ⟨Î⟩
I_vacuum = I_total(phases_vacuum)
I_tau = I_total(phases_tau)
ΔI = I_tau - I_vacuum

# 6. Wyznaczenie masy cząstki
mass_tau_MeV = ΔI * Λ

# 7. Wynik i wykres
print(f"⟨Î⟩ próżni: {I_vacuum:.4f}")
print(f"⟨Î⟩ tau: {I_tau:.4f}")
print(f"Δ⟨Î⟩ = {ΔI:.4f}")
print(f"m_tau c² ≈ {mass_tau_MeV:.2f} MeV")

plt.bar(["Próżnia", "Tau"], [I_vacuum, I_tau], color=["blue", "red"])
plt.ylabel("⟨Î⟩")
plt.title("Operator dyssynchronizacji w TST")
plt.grid(True)
plt.show()
