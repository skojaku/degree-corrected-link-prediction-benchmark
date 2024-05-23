import unittest
from dclinkpred import LinkPredictionDataset
import networkx as nx
import numpy as np

class TestLinkPredictionDataset(unittest.TestCase):
    def setUp(self):
        G = nx.karate_club_graph()
        self.num_edges = G.number_of_edges()
        self.adj_matrix = nx.adjacency_matrix(G)
        self.G = G
        #self.num_edges = self.adj_matrix.nnz / 2

    def test_initialization(self):
        model = LinkPredictionDataset(
            testEdgeFraction=0.2, degree_correction=True,
        )
        self.assertIsNotNone(model)
        model = LinkPredictionDataset(
            testEdgeFraction=0.2,
            degree_correction=False
        )
        self.assertIsNotNone(model)

    def test_fit_transform(self):

        testEdgeFraction = 0.2
        for degree_correction in [True, False]:
            for net in [self.G, self.adj_matrix]:
                model = LinkPredictionDataset(
                    testEdgeFraction=testEdgeFraction,
                    degree_correction=degree_correction,
                )
                model.fit(net)
                train_net, src_test, trg_test, y_test = model.transform()

                # Check if the train network has fewer edges due to test edge removal
                if isinstance(train_net, nx.Graph):
                    self.assertTrue(train_net.number_of_edges() < self.num_edges)
                else:
                    self.assertTrue(train_net.nnz *0.5 < self.num_edges)

                # Check if the number of positive edges in target table matches expected test edges
                num_test_edges = int(self.num_edges * testEdgeFraction)
                num_positives = np.sum(y_test == 1)
                num_negatives = np.sum(y_test == 0)
                self.assertEqual(num_positives, num_test_edges)

                # Check if the number of negative samples is approximately equal to the number of positive samples
                self.assertEqual(num_positives, num_negatives)
            model = LinkPredictionDataset(
                testEdgeFraction=testEdgeFraction,
                degree_correction=degree_correction,
            )
            model.fit(self.adj_matrix)
            train_net, src_test, trg_test, y_test = model.transform()

            # Check if the train network has fewer edges due to test edge removal
            if isinstance(train_net, nx.Graph):
                self.assertTrue(train_net.number_of_edges() < self.num_edges)
            else:
                self.assertTrue(train_net.nnz *0.5 < self.num_edges)

            # Check if the number of positive edges in target table matches expected test edges
            num_test_edges = int(self.num_edges * testEdgeFraction)
            num_positives = np.sum(y_test == 1)
            num_negatives = np.sum(y_test == 0)
            self.assertEqual(num_positives, num_test_edges)

            # Check if the number of negative samples is approximately equal to the number of positive samples
            self.assertEqual(num_positives, num_negatives)

if __name__ == '__main__':
    unittest.main()
