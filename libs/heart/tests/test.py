# %%
import networkx as nx
import heart
import numpy as np
from scipy import sparse

G = nx.karate_club_graph()
A = nx.adjacency_matrix(G)
labels = np.unique([d[1]['club'] for d in G.nodes(data=True)], return_inverse=True)[1]

sampler = heart.HeartSampler(testEdgeFraction = 0.1)
sampler.generate_samples(sparse.csr_matrix(A))
