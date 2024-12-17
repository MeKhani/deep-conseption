import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tool import load_entites_dic
from tool import read_triplet
from tool import divid_entities
from tool import exrtract_relation_in_types_entities
from tool import create_entity_type_triple
from tool import create_graph
from tool import add_feature_to_graph_nodes
from tool import add_feature_to_graph_edges
from model import GraphSAGE
from model import GCNWithEdgeFeatures
from model import EdgeFeatureGNNLayer
from model import GraphAutoencoder
from model import loss_function
from model import generat_emmbedding_by_random_walk
# import loadMkg as lmkg
import pandas as pd
from my_parser import parse


# from utils.utils_train import encoding_train
# from utils.accuracy import compute_accuracy_for_train
# from GNN.model import GCN


import time

args = parse()

#parse the argument

print("I'm here ")
# print("graph name",args.name_graph)

np.random.seed(1)
torch.manual_seed(1)
#load fb15k237
#read nell data base 
path = args.data_path + args.data_name + "/"+"train.txt"
# dir  = "dataset/NELL-995-v1/train.txt"

id2ent , id2rel ,triple_entities , en2id, rel2id = read_triplet(path)
endic , en_dic_id= divid_entities(id2ent,en2id)
# create_triples_for_type_entities(triple_entities, en_dic_id)
# print(en_dic_id)

print("dic length" , len(endic))
# print("relation to id   " , rel2id)
# print("id to relation   " , id2rel)
num_relations = len(id2rel)
args.num_rel = num_relations
# print(en2id)
# print(en_dic_id)
i_r  ,o_r_c,o_r_s= exrtract_relation_in_types_entities(triple_entities,en_dic_id)
# print("recived_entity",o_r_c)
# print("inner relations ",i_r)
entity_type_triple = create_entity_type_triple(o_r_s,o_r_c)
graph = create_graph(entity_type_triple)
# print(en_dic_id)
graph = add_feature_to_graph_nodes(graph,i_r,en_dic_id, num_relations)
graph = add_feature_to_graph_edges(graph,entity_type_triple ,  num_relations)

print(graph.edata["rel_feat"][0])
print(graph.edata["rel_feat"][145])
print(graph.edata["rel_feat"][5])
#define model 
node_features = graph.ndata["feat"]
edge_features = graph.edata["rel_feat"]
model =GraphSAGE(in_feats=num_relations+1,hidden_feats=32,out_feats=32) 
optimizer = Adam(model.parameters(),lr=args.learning_rate)
for epoch in range(args.num_epoch):
    model.train()
    #forward pass 
    node_embeddings= model(graph,node_features)
    # Example loss: Contrastive loss (random pairs)

    loss = torch.mean(node_embeddings)  # Replace with appropriate unsupervised loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
print(node_embeddings)
model1 =   GraphAutoencoder(
        node_in_feats=num_relations+1,
        edge_in_feats=num_relations,
        hidden_feats=64,
        latent_feats=32
    )
optimizer = Adam(model1.parameters(),lr=args.learning_rate)
# True adjacency matrix
adj_true = graph.adjacency_matrix().to_dense()
for epoch in range(args.num_epoch):
    model1.train()
    optimizer.zero_grad()
   # Forward pass
    _, adj_pred = model1(graph, node_features, edge_features)
    
    # Compute loss
    loss = loss_function(adj_pred, adj_true)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{args.num_epoch}, Loss: {loss.item():.4f}")













#encoding the data
#load some data for examples 
