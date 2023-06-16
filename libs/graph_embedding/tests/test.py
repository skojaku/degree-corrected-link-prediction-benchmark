import unittest

import networkx as nx
import numpy as np
from scipy import sparse

import graph_embedding


class TestCalc(unittest.TestCase):
    def setUp(self):
        self.G = nx.karate_club_graph()
        self.A = nx.adjacency_matrix(self.G)

    def test_mod_spectral(self):
        model = graph_embedding.embeddings.ModularitySpectralEmbedding()
        model.fit(self.A)
        vec = model.transform(dim=32)

    def test_adj_spectral(self):
        model = graph_embedding.embeddings.AdjacencySpectralEmbedding()
        model.fit(self.A)
        vec = model.transform(dim=32)

    def test_leigenmap(self):
        model = graph_embedding.embeddings.LaplacianEigenMap()
        model.fit(self.A)
        vec = model.transform(dim=32)

    def test_node2vec(self):
        model = graph_embedding.embeddings.Node2Vec()
        model.fit(self.A)
        vec = model.transform(dim=32)

    def test_deepwalk(self):
        model = graph_embedding.embeddings.DeepWalk()
        model.fit(self.A)
        vec = model.transform(dim=32)

    def test_nonbacktracking(self):
        model = graph_embedding.embeddings.NonBacktrackingSpectralEmbedding()
        model.fit(self.A)
        vec = model.transform(dim=32)

    def test_node2vec_matrix_factorization(self):
        model = graph_embedding.embeddings.Node2VecMatrixFactorization()
        model.fit(self.A)
        vec = model.transform(dim=32)

    def test_highorder_modularity_spec_embedding(self):
        model = graph_embedding.embeddings.HighOrderModularitySpectralEmbedding()
        model.fit(self.A)
        vec = model.transform(dim=32)

    def test_normalized_trans_matrix_spec_embedding(self):
        model = graph_embedding.embeddings.LinearizedNode2Vec()
        model.fit(self.A)
        vec = model.transform(dim=32)

    def test_non_backtracking_node2vec(self):
        model = graph_embedding.embeddings.NonBacktrackingNode2Vec()
        model.fit(self.A)
        vec = model.transform(dim=32)

    def test_non_backtracking_deepwalk(self):
        model = graph_embedding.embeddings.NonBacktrackingDeepWalk()
        model.fit(self.A)
        vec = model.transform(dim=32)

    def test_torch_node2vec(self):
        model = graph_embedding.TorchNode2Vec()
        model.fit(self.A)
        vec = model.transform(dim=32)

    def test_torch_node2vec_linear(self):
        model = graph_embedding.TorchModularityFactorization()
        model.fit(self.A)
        vec = model.transform(dim=32)

if __name__ == "__main__":
    unittest.main()
