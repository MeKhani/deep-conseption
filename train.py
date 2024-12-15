import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tool import load_entites_dic
from tool import read_triplet
from tool import divid_entities
from tool import exrtract_relation_in_types_entities
from tool import create_entity_type_triple
from tool import create_graph
from tool import add_feature_to_graph
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
add_feature_to_graph(graph,i_r,en_dic_id)
# graph.ndata["feat"]= torch.ones(graph.num_nodes(),num_relations+1,2)
# print("Node features")
# print(graph.ndata["feat"][1])
# for node in graph.nodes():
#     print(graph.ndata['feat'][node] )

    # print(f"Node ID: {node}, Feature: {graph.ndata['feature'][node]}")
# print(graph)

# print("inner realtion in every type",entity_type_triple)
# print("inner realtion in every type",i_r)
# print("outer recived relation ",o_r_s)
# array_triples = np.array(triple_entities)
# print(array_triples[])

# for key , value in endic.items() :
#     print("keys is ",key , "values : " , value ,"\n ")

# print(entities) 







#encoding the data
#load some data for examples 
