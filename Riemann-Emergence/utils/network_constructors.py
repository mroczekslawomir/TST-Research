"""
Helper functions for constructing networks used in TST simulations
"""

import networkx as nx

def create_ws_network(N=1000, k=4, p=0.1, seed=42):
    return nx.watts_strogatz_graph(N, k, p, seed=seed)

def create_ba_network(N=1000, m=3, seed=42):
    return nx.barabasi_albert_graph(N, m, seed=seed)

def create_er_network(N=1000, p=0.01, seed=42):
    return nx.erdos_renyi_graph(N, p, seed=seed)
