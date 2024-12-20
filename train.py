import os
import numpy as np
import torch
from torch.optim import Adam
import pandas as pd
from tqdm import tqdm

from tool import (
    load_entites_dic,
    read_triplet,
    divid_entities,
    exrtract_relation_in_types_entities,
    create_entity_type_triple,
    create_graph,
    add_feature_to_graph_nodes,
    add_feature_to_graph_edges,
    exrtract_relation_in_types_entities1,
    generate_group_triples ,
)
from model import (
    GraphSAGE,
    GraphAutoencoder,
    GraphAutoencoderGNN,
    loss_function,
)
from my_parser import parse


def set_seed(seed=1):
    """Set the seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(args):
    """
    Load and process graph data from the provided path.
    
    Returns:
        graph (DGLGraph): Processed graph with features.
        num_relations (int): Number of relations in the dataset.
    """
    print("Loading data...")
    path = os.path.join(args.data_path, args.data_name, "train.txt")

    # Read and process triplets
    id2ent, id2rel, triplets, en2id, rel2id = read_triplet(path)
    endic, en_dic_id = divid_entities(id2ent, en2id)
    print(len(en2id))

    print("Entity dictionary length:", len(endic))

    # Extract relations and create graph
   
    entity_type_triples ,inner_rel ,output_relations,input_relations = generate_group_triples(en_dic_id,triplets)
   
    graph = create_graph(entity_type_triples)

    # Add features to the graph
    num_relations = len(id2rel)
    print("number relations:", num_relations)
    graph = add_feature_to_graph_nodes(graph, inner_rel,output_relations,input_relations, num_relations)
    graph = add_feature_to_graph_edges(graph, entity_type_triples, num_relations)
   
    print("Sample edge features:", graph.edata["rel_feat"][0])

    return graph, num_relations


def train_model(model, optimizer, graph, node_features, edge_features=None, adj_true=None, loss_fn=None, epochs=10):
    """
    Generic training loop for GNN models.

    Args:
        model (nn.Module): GNN model to train.
        optimizer (Optimizer): Optimizer for the model.
        graph (DGLGraph): Input graph.
        node_features (Tensor): Node feature tensor.
        edge_features (Tensor, optional): Edge feature tensor.
        adj_true (Tensor, optional): Ground truth adjacency matrix.
        loss_fn (callable, optional): Loss function to use.
        epochs (int): Number of training epochs.

    Returns:
        embeddings (Tensor): Learned node embeddings.
    """
    print(f"Training {model.__class__.__name__}...")
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        # Forward pass
        if edge_features is not None:
            # Models that expect edge features
            outputs = model(graph, node_features, edge_features)
        else:
            # Models like GraphSAGE that don't need edge features
            outputs = model(graph, node_features)

        # Handle models with one or two outputs
        if isinstance(outputs, tuple):
            embeddings, adj_pred = outputs
        else:
            embeddings = outputs
            adj_pred = None

        # Compute loss
        if loss_fn and adj_true is not None and adj_pred is not None:
            loss = loss_fn(adj_pred, adj_true)
        else:
            loss = torch.mean(embeddings)  # Placeholder loss for unsupervised training

        # Backward and optimize
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    return embeddings
def main():
    """Main entry point of the program."""
    # Parse arguments and set seed
    args = parse()
    set_seed()

    # Load data
    graph, num_relations = load_data(args)
    node_features = graph.ndata["feat"]
    edge_features = graph.edata["rel_feat"]
    adj_true = graph.adjacency_matrix().to_dense()

    # Update args with the number of relations
    args.num_rel = num_relations
    # print(node_features.shape[1])

    # Train GraphSAGE
    sage_model = GraphSAGE(in_feats=node_features.shape[1], hidden_feats=32, out_feats=32)
    sage_optimizer = Adam(sage_model.parameters(), lr=args.learning_rate)
    sage_embeddings = train_model(sage_model, sage_optimizer, graph, node_features, epochs=args.num_epoch)
    print("GraphSAGE embeddings:", sage_embeddings)

    # Train GraphAutoencoder
    gae_model = GraphAutoencoder(node_in_feats=node_features.shape[1], edge_in_feats=edge_features.shape[1], hidden_feats=64, latent_feats=32)
    gae_optimizer = Adam(gae_model.parameters(), lr=args.learning_rate)
    gae_embeddings = train_model(
        gae_model,
        gae_optimizer,
        graph,
        node_features,
        edge_features,
        adj_true,
        loss_function,
        epochs=args.num_epoch,
    )
    print("GraphAutoencoder embeddings:", gae_embeddings)

    # Train GraphAutoencoderGNN
    gnn_gae_model = GraphAutoencoderGNN(node_features.shape[1], 32, 32)
    gnn_gae_optimizer = Adam(gnn_gae_model.parameters(), lr=args.learning_rate)
    gnn_gae_embeddings = train_model(
        gnn_gae_model,
        gnn_gae_optimizer,
        graph,
        node_features,
        adj_true=adj_true,
        loss_fn=loss_function,
        epochs=args.num_epoch,
    )
    print("GraphAutoencoderGNN embeddings:", gnn_gae_embeddings)

main()

# if __name__ == "__main__":
#     main()
