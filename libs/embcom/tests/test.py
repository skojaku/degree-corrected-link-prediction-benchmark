# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-03-30 09:38:48
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-04-11 09:28:38
import unittest

import networkx as nx
import numpy as np
from scipy import sparse

import embcom

import networkx as nx
import numpy as np
from scipy import sparse
import embcom 
import unittest


def inheritors(klass):
    subclasses = set()
    work = [klass]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                work.append(child)
    return subclasses


class TestingEmbeddingMethods(unittest.TestCase):
    def setUp(self):
        G = nx.karate_club_graph()
        self.A = nx.adjacency_matrix(G)
        self.labels = np.unique(
            [d[1]["club"] for d in G.nodes(data=True)], return_inverse=True
        )[1]

    def test_embedding(self):
        for model in inheritors(embcom.NodeEmbeddings):
            instance = model()
            instance.fit(self.A)
            instance.transform(dim=8)
            print(model)
