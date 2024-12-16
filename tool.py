import numpy as np
import scipy.sparse as sp
import torch
import time
import random
import dgl
import os


def read_data(path):
    
    """read file and return the triples containing its ground truth label (0/1)"""
    
    f = open(path)
    triples_with_label = []
    for line in f:
        triple_with_label = line.strip().split("\t")
        triples_with_label.append(triple_with_label)
    f.close()
    return triples_with_label

def write_dic(write_path, array):
    
    """generate a dictionary"""
    
    f = open(write_path, "w+")
    for i in range(len(array)):
        f.write("{}\t{}\n".format(i, array[i]))
    f.close()
    print("saved dictionary to {}".format(write_path))
    
def dictionary(input_list):
    
    """
    To generate a dictionary.
    Index: item in the array.
    Value: the index of this item.
    """
    
    return dict(zip(input_list, range(len(input_list))))

def normalize(mx):
    
    """Row-normalize sparse matrix"""
    
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def load_entites_dic(path ):
    entities = {}

    with open(path, 'r') as file:
        for line in file:
            # Split each line into ID and label
            entity_id, entity_label = line.strip().split()
            # Add to dictionary
            entities[entity_id] = entity_label
    return entities


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def split_known(triples):
    
    """
    Further split the triples into 2 sets:
    1. an incomplete graph: known
    2. a set of missing facts we want to recover: unknown
    """
    
    DATA_LENGTH = len(triples)
    split_ratio = [0.9, 0.1]
    candidate = np.array(range(DATA_LENGTH))
    np.random.shuffle(candidate)
    idx_known = candidate[:int(DATA_LENGTH * split_ratio[0])]
    idx_unknown = candidate[int(DATA_LENGTH * split_ratio[0]):]
    known = []
    unknown = []
    for i in idx_known:
        known.append(triples[i])
    for i in idx_unknown:
        unknown.append(triples[i])
    return known, unknown
def read_triplet(path):
            
        
       
            id2ent, id2rel, triplets = [], [], []
            rel_info = {}
            pair_info = {}
            with open(path, 'r') as f:
                for line in f.readlines():
                    h, r, t = line.strip().split('\t')
                    id2ent.append(h)
                    id2ent.append(t)
                    id2rel.append(r)
                    triplets.append((h, r, t))
            id2ent = remove_duplicate(id2ent)
            id2rel = remove_duplicate(id2rel)
            ent2id = {ent: idx for idx, ent in enumerate(id2ent)}
            rel2id = {rel: idx for idx, rel in enumerate(id2rel)}
            triplets = [(ent2id[h], rel2id[r], ent2id[t]) for h, r, t in triplets]
            for (h,r,t) in triplets:
                if (h,t) in rel_info:
                    rel_info[(h,t)].append(r)
                else:
                    rel_info[(h,t)] = [r]
                if r in pair_info:
                    pair_info[r].append((h,t))
                else:
                    pair_info[r] = [(h,t)]
            # G = igraph.Graph.TupleList(np.array(triplets)[:, 0::2])
            # G_ent = igraph.Graph.TupleList(np.array(triplets)[:, 0::2], directed = True)
            # spanning = G_ent.spanning_tree()
            # G_ent.delete_edges(spanning.get_edgelist())
            
            # for e in spanning.es:
            #     e1,e2 = e.tuple
            #     e1 = spanning.vs[e1]["name"]
            #     e2 = spanning.vs[e2]["name"]
            #     self.spanning.append((e1,e2))
            
            # spanning_set = set(self.spanning)


            
            # print("-----Train Data Statistics-----")
            # print(f"{len(self.ent2id)} entities, {len(self.rel2id)} relations")
            # print(f"{len(triplets)} triplets")
            # self.triplet2idx = {triplet:idx for idx, triplet in enumerate(triplets)}
            # self.triplets_with_inv = np.array([(t, r + len(id2rel), h) for h,r,t in triplets] + triplets)
            return id2ent, id2rel, triplets,ent2id, rel2id
def remove_duplicate(x):
	    return list(dict.fromkeys(x))
def divid_entities(entities , en2id ):
     dicOfEntities={}
     dicOfEntitiesIds={}
     count = -1
     for en , val in en2id.items():
          consept, typeOfEn, en1= en.strip().split(':')
          if typeOfEn in dicOfEntities:
                    dicOfEntities[typeOfEn].append(en1)
                    dicOfEntitiesIds[count].append(val)
          else:
                count +=1
                dicOfEntities[typeOfEn]  = [en]
                dicOfEntitiesIds[count]  = [val]
    #  print (typeOfEn+"\n")
     return dicOfEntities , dicOfEntitiesIds
def exrtract_relation_in_types_entities(triple_entities, en_dic_id):
    inner_relations_for_every_type = {}
    outer_recived_relations_for_every_type = {}
    outer_sent_relations_for_every_type = {}

    for key, val in en_dic_id.items():
        for en in val:
            # Collect unique relations where `en` is a subject or object
            inner_relations_subject = {relation for subj, relation, obj in triple_entities if subj == en and obj in val}
            outer_relations_subject = {relation for subj, relation, obj in triple_entities if subj == en and obj not in  val}
            inner_relations_object =  {relation for subj, relation, obj in triple_entities if obj == en and subj in val}
            outer_relations_object =  {relation for subj, relation, obj in triple_entities if obj == en and subj not in val}

            # Update received relations dictionary
            if inner_relations_subject:
                if key not in inner_relations_for_every_type:
                    inner_relations_for_every_type[key] = inner_relations_subject 
                else:
                    inner_relations_for_every_type[key].update(inner_relations_subject) 
            if outer_relations_subject:
                if key not in outer_recived_relations_for_every_type:
                    outer_recived_relations_for_every_type[key] = outer_relations_subject 
                else:
                    outer_recived_relations_for_every_type[key].update(outer_relations_subject)
            
            # Update sent relations dictionary
            if inner_relations_object:
                if key not in inner_relations_for_every_type:
                    inner_relations_for_every_type[key] = inner_relations_object 
                else:
                    inner_relations_for_every_type[key].update(inner_relations_object)
            if outer_relations_object:
                if key not in outer_sent_relations_for_every_type:
                    outer_sent_relations_for_every_type[key] = outer_relations_object 
                else:
                    outer_sent_relations_for_every_type[key].update(outer_relations_object)
    

    return inner_relations_for_every_type,outer_recived_relations_for_every_type,outer_sent_relations_for_every_type
def create_entity_type_triple(outer_sent_relations_for_every_type,outer_recived_relations_for_every_type):
     entity_type_triples = []
     for obj , relations in outer_sent_relations_for_every_type.items():
          for rel in relations:
               subj_ent_types = [en_type for en_type, val in outer_recived_relations_for_every_type.items() if rel in val  ]
               for subj  in subj_ent_types:
                    entity_type_triples.append((obj,rel,subj))
                    
          
     
     return entity_type_triples
def create_graph(triples):
     # Extract node and edge information
    src_nodes = [t[0] for t in triples]  # subjects
    dst_nodes = [t[2] for t in triples]  # objects
    edge_types = [t[1] for t in triples]  # relations

    # Create a DGL graph
    g = dgl.heterograph({
        ('node', 'relation_type', 'node'): (src_nodes, dst_nodes)
    })
    
    return g
def add_feature_to_graph_nodes(graph, i_r, en_dic_id, num_rel):
    # Initialize node features with zeros
    graph.ndata["feat"] = torch.zeros(graph.num_nodes(), num_rel + 1)
    
    # Iterate over nodes and update their features
    for node, features in enumerate(graph.ndata["feat"]):
        # Set the 0th feature to the length of en_dic_id[node] (default to 0 if not present)
        features[0] = len(en_dic_id.get(node, []))
        
        # Update relation-based features if the node exists in i_r
        if node in i_r:
            for v in i_r[node]:
                features[v + 1] = 1

    return graph

def add_feature_to_graph_edges(graph, entity_type_triples, num_relations):
    # Initialize edge features with zeros
    rel_feat = torch.zeros(graph.num_edges(), num_relations)
    
    # Create a lookup set for quick membership testing
    entity_triples_set = set(entity_type_triples)
    
    # Get all edges
    src_nodes, dst_nodes = graph.edges()
    
    # Iterate through relations and update features
    for rel in range(num_relations):
        mask = [
            (src.item(), rel, dst.item()) in entity_triples_set
            for src, dst in zip(src_nodes, dst_nodes)
        ]
        rel_feat[torch.tensor(mask), rel] = 1

    # Assign edge features
    graph.edata["rel_feat"] = rel_feat

    return graph

