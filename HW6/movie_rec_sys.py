from collections import deque
# credit https://github.com/networkx/networkx/blob/master/networkx/algorithms/centrality/betweenness.py
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

G = nx.Graph()
DATA_DIR = '/Users/aman/Dropbox/CS6220_Amandeep_Singh/HW6/edges_sampled_2K.csv'

edges = np.loadtxt(fname=DATA_DIR, delimiter=',').astype('int')
for e in edges.tolist():
    G.add_edge(*e)


def single_source_shortest_path_basic(G, s):
    S = []
    P = {}
    for v in G:
        P[v] = []
    sigma = dict.fromkeys(G, 0.0)  # sigma[v]=0 for v in G
    D = {}
    sigma[s] = 1.0
    D[s] = 0
    Q = [s]
    while Q:  # use BFS to find shortest paths
        v = Q.pop(0)
        S.append(v)
        Dv = D[v]
        sigmav = sigma[v]
        for w in G[v]:
            if w not in D:
                Q.append(w)
                D[w] = Dv + 1
            if D[w] == Dv + 1:  # this is a shortest path, count paths
                sigma[w] += sigmav
                P[w].append(v)  # predecessors
    return S, P, sigma


def rescale_e(betweenness, n):
    if n <= 1:
        scale = None  # no normalization b=0 for all nodes
    else:
        scale = 1.0 / (n * (n - 1))
    if scale is not None:
        for v in betweenness:
            betweenness[v] *= scale
    return betweenness


def _accumulate_edges(betweenness, S, P, sigma, s):
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1.0 + delta[w]) / sigma[w]
        for v in P[w]:
            c = sigma[v] * coeff
            if (v, w) not in betweenness:
                betweenness[(w, v)] += c
            else:
                betweenness[(v, w)] += c
            delta[v] += c
        if w != s:
            betweenness[w] += delta[w]
    return betweenness


def betweeness(G):
    betweenness = dict.fromkeys(G, 0.0)  # b[v]=0 for v in G
    # b[e]=0 for e in G.edges()
    betweenness.update(dict.fromkeys(G.edges(), 0.0))
    nodes = G
    for s in nodes:
        S, P, sigma = single_source_shortest_path_basic(G, s)
        betweenness = _accumulate_edges(betweenness, S, P, sigma, s)
    # rescaling
    for n in G:  # remove nodes to only return edges
        del betweenness[n]
    betweenness = rescale_e(betweenness, len(G))
    return betweenness


def edge_to_remove(G):
    dict_b = betweeness(G)
    list_of_tup = dict_b.items()
    list_of_tup.sort(key=lambda x: x[1], reverse=True)
    return list_of_tup[0][0]


def girvan(G):
    plt.figure()
    nx.draw(G)
    plt.show()
    c = [_ for _ in nx.connected_component_subgraphs(G)]
    l = len(c)
    while l < 10:
        print("Number of connected components {0}".format(str(l)))
        G.remove_edge(*edge_to_remove(G))
        c = [_ for _ in nx.connected_component_subgraphs(G)]
        l = len(c)
    for _ in [g.nodes() for g in c]:
        print(_)


girvan(G)
