# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-02 20:44:01
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-10-02 21:39:59
from .node2vec import Node2Vec
from .models import Word2Vec
from .dataset import NegativeSamplingDataset, ModularityEmbeddingDataset
from .loss import TripletLoss, ModularityTripletLoss, DistanceMetrics
from .train import train


class TorchNode2Vec(Node2Vec):
    def __init__(self, device="cpu", batch_size=1024, logsigmoid_loss=True, **params):
        super().__init__(**params)
        self.device = device
        self.batch_size = batch_size
        self.logsigmoid_loss = logsigmoid_loss

    def update_embedding(self, dim):

        # Word2Vec model
        model = Word2Vec(n_nodes=self.num_nodes, dim=dim)

        # Set up negative sampler
        dataset = NegativeSamplingDataset(
            seqs=self.sampler,
            window=self.window,
            epochs=self.epochs,
            context_window_type="double",
            num_negative_samples=self.negative,
            ns_exponent=self.ns_exponent,
        )

        # Set up the loss function
        loss_func = TripletLoss(
            model,
            dist_metric=DistanceMetrics.DOTSIM,
            logsigmoid_loss=self.logsigmoid_loss,
        )

        # Train
        train(
            model=model,
            dataset=dataset,
            loss_func=loss_func,
            batch_size=self.batch_size,
            device=self.device,
            learning_rate=self.alpha,
        )
        self.in_vec = model.embedding()
        self.out_vec = model.embedding(return_out_vector=True)


class TorchModularityFactorization(Node2Vec):
    def __init__(self, device="cpu", batch_size=1024, logsigmoid_loss=True, **params):
        super().__init__(**params)
        self.device = device
        self.batch_size = batch_size

    def update_embedding(self, dim):

        # Word2Vec model
        model = Word2Vec(n_nodes=self.num_nodes, dim=dim)

        # Set up negative sampler
        dataset = ModularityEmbeddingDataset(
            seqs=self.sampler,
            window=self.window,
            epochs=self.epochs,
            context_window_type="double",
            num_negative_samples=self.negative,
            ns_exponent=self.ns_exponent,
        )

        # Set up the loss function
        loss_func = ModularityTripletLoss(
            model,
            dist_metric=DistanceMetrics.DOTSIM,
        )

        # Train
        train(
            model=model,
            dataset=dataset,
            loss_func=loss_func,
            batch_size=self.batch_size,
            device=self.device,
            learning_rate=self.alpha,
        )
        self.in_vec = model.embedding()
        self.out_vec = model.embedding(return_out_vector=True)
