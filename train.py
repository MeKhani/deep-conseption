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
# import loadMkg as lmkg
import pandas as pd


# from utils.utils_train import encoding_train
# from utils.accuracy import compute_accuracy_for_train
# from GNN.model import GCN


import time


#parse the argument

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, required=True,
                    help='Name of training dataset')
parser.add_argument('--epoch', type=int, default=1000,
                    help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate of the optimizer')
parser.add_argument('--weight_decay', type=float, default=5e-8,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Dimension of hidden vectors')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate')
# parser.add_argument('--dataset_path',type=str ,default="dataset/NELL-995-v1",help="" )

args = parser.parse_args()
print("I'm here ")

np.random.seed(1)
torch.manual_seed(1)
#load fb15k237
#read nell data base 
dir  = "dataset/NELL-995-v1/train.txt"

id2ent , i1 ,triple_entities , en2id= read_triplet(dir)
endic , en_dic_id= divid_entities(id2ent,en2id)
# create_triples_for_type_entities(triple_entities, en_dic_id)
print("dic length" , len(endic))
# print(en2id)
# print(en_dic_id)
i_r  ,o_r_c,o_r_s= exrtract_relation_in_types_entities(triple_entities,en_dic_id)
# print("recived_entity",o_r_c)
entity_type_triple = create_entity_type_triple(o_r_s,o_r_c)

print("inner realtion in every type",entity_type_triple)
# print("inner realtion in every type",i_r)
# print("outer recived relation ",o_r_s)
# array_triples = np.array(triple_entities)
# print(array_triples[])

# for key , value in endic.items() :
#     print("keys is ",key , "values : " , value ,"\n ")

# print(entities) 







#encoding the data
#load some data for examples 
