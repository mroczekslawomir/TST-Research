"""
Construction of the desynchronization operator Î in TST
"""
import numpy as np
import networkx as nx
from scipy.sparse import diags, identity

def build_desynchronization_operator(G, A=1.0, B=1.0, C=0.1):
    n = G.number_of_nodes()
    L = nx.laplacian_matrix(G).astype(float)
    I_sync = identity(n, format='csr') * A
    degrees = np.array([d for _, d in G.degree()])
    I_ent = diags(C * degrees, 0, format='csr')
    return B * L + I_sync + I_ent
