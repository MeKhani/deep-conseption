import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv
from dgl.nn import GraphConv, EdgeWeightNorm

class GCNWithEdgeFeatures(nn.Module):
    def __init__(self, in_feats, edge_feats, hidden_feats, out_feats):
        super(GCNWithEdgeFeatures, self).__init__()
        self.edge_norm = EdgeWeightNorm(norm='both')  # Normalize edge weights
        self.conv1 = GraphConv(in_feats, hidden_feats, allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_feats, out_feats, allow_zero_in_degree=True)
        self.edge_fc = nn.Linear(edge_feats, 1)  # Map edge features to a scalar weight

    def forward(self, graph, node_feats, edge_feats):
        # Compute edge weights from edge features
        edge_weights = self.edge_fc(edge_feats).squeeze(-1)
        edge_weights = self.edge_norm(graph, edge_weights)
        
        # First GCN layer
        h = self.conv1(graph, node_feats, edge_weight=edge_weights)
        h = F.relu(h)

        # Second GCN layer
        h = self.conv2(graph, h, edge_weight=edge_weights)
        return h

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_feats, 'mean')
        self.conv2 = SAGEConv(hidden_feats, out_feats, 'mean')

    def forward(self, graph, node_feats):
        h = self.conv1(graph, node_feats)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h

