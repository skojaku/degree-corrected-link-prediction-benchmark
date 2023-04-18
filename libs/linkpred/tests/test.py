# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-04-11 16:31:47
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-04-11 16:34:43
import unittest
import networkx as nx
import numpy as np
from scipy import sparse
from linkpred.LinkPredictionDataset import LinkPredictionDataset


class TestLinkPred(unittest.TestCase):
    def setUp(self):
        self.G = nx.karate_club_graph()
        self.A = nx.adjacency_matrix(self.G)

    def test_link_prediction_dataset(self):
        for sampler in ["uniform", "degreeBiased"]:
            model = LinkPredictionDataset(
                negative_edge_sampler=sampler,
                testEdgeFraction=0.5,
                conditionedOnSource=True,
            )
            model.fit(self.A)
            train_net, edge_table = model.transform()


if __name__ == "__main__":
    unittest.main()
