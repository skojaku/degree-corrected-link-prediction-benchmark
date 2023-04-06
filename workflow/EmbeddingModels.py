# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-14 15:08:01
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-03-31 17:44:39

import embcom
import torch
import numpy as np

embedding_models = {}
embedding_model = lambda f: embedding_models.setdefault(f.__name__, f)


@embedding_model
def node2vec(network, dim, window_length=10, num_walks=40):
    model = embcom.embeddings.Node2Vec(window_length=window_length, num_walks=num_walks)
    model.fit(network)
    return model.transform(dim=dim)


@embedding_model
def deepwalk(network, dim, window_length=10, num_walks=40):
    model = embcom.embeddings.DeepWalk(window_length=window_length, num_walks=num_walks)
    model.fit(network)
    return model.transform(dim=dim)


##Laplacian Eigen map ==> make a feature 
## 

@embedding_model
def leigenmap(network, dim):
    model = embcom.embeddings.LaplacianEigenMap()
    model.fit(network)
    return model.transform(dim=dim)


@embedding_model
def modspec(network, dim):
    model = embcom.embeddings.ModularitySpectralEmbedding()
    model.fit(network)
    return model.transform(dim=dim)


@embedding_model
def GCN(network,dim,feature_dim=10,device='cpu',dim_h=128):
    
    """
    Parameters
    ----------
    network: adjacency matrix
    feature_dim: dimension of features
    dim: dimension of embedding vectors
    dim_h : dimension of hidden layer
    device : device

    """
    
    model = embcom.embeddings.LaplacianEigenMap()
    model.fit(network)
    features = model.transform(dim=feature_dim)

    
    model_GCN, data = embcom.embeddings.GCN(feature_dim,dim_h,dim).to(device), torch.from_numpy(features).to(dtype=torch.float,device = device)
    model_trained = embcom.train(model_GCN,data,network,device)

    network_c = network.tocoo()
    
    edge_list_gcn = torch.from_numpy(np.array([network_c.row, network_c.col])).to(device)
    
    

    embeddings = model_trained(data,edge_list_gcn)
    
    return embeddings


def GAN(network,dim,feature_dim=10,device='cpu',dim_h=128):
    
    """
    Parameters
    ----------
    network: adjacency matrix
    feature_dim: dimension of features
    dim: dimension of embedding vectors
    dim_h : dimension of hidden layer
    device : device

    """
    
    model = embcom.embeddings.LaplacianEigenMap()
    model.fit(network)
    features = model.transform(dim=feature_dim)

    
    model_GAN, data = embcom.embeddings.GAN(feature_dim,dim_h,dim).to(device), torch.from_numpy(features).to(dtype=torch.float,device = device)
    model_trained = embcom.train(model_GAN,data,network,device)

    network_c = network.tocoo()
    
    edge_list_gan = torch.from_numpy(np.array([network_c.row, network_c.col])).to(device)
    
    

    embeddings = model_trained(data,edge_list_gan)
    
    return embeddings



# @embedding_model
# def nonbacktracking(network, dim):
#    model = embcom.embeddings.NonBacktrackingSpectralEmbedding()
#    model.fit(network)
#    return model.transform(dim=dim)
