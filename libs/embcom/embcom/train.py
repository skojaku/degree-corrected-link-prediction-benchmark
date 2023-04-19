import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from torch_geometric.utils import negative_sampling
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv

from torch_geometric.utils import train_test_split_edges

import numpy as np
from tqdm import tqdm




def get_link_labels(pos_edge_index, neg_edge_index,device):
    # returns a tensor:
    # [1,1,1,1,...,0,0,0,0,0,..] with the number of ones is equel to the lenght of pos_edge_index
    # and the number of zeros is equal to the length of neg_edge_index
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels






def train(model,data,adjacency,device):
    
    
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

    model.train()
    
    epochs = 2000
    
    adj_c = adjacency.tocoo()
    
    edge_list_torch = torch.from_numpy(np.array([adj_c.row, adj_c.col])).to(device)
    
    
    
    for epoch in tqdm(range(epochs+1)):
   
        neg_edge_index = negative_sampling(
            
        edge_index=  edge_list_torch, #positive edges
        num_nodes = data.shape[0], # number of nodes
        num_neg_samples= edge_list_torch.size(1))
        
        optimizer.zero_grad()
        
        z = model(data,edge_list_torch)
        
        link_logits = model.decode(z, edge_list_torch, neg_edge_index) # decode
        link_labels = get_link_labels(edge_list_torch, neg_edge_index,device)
        loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
        loss.backward()
        optimizer.step()
       

        
    return model
        
        
