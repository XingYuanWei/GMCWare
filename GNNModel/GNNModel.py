import torch
import torch.nn as nn
import dgl
import dgl.nn.pytorch as dglnn


# GCN Model
class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GCNModel, self).__init__()
        self.conv1 = dglnn.GraphConv(input_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, num_classes)

    def forward(self, g, h):
        h = self.conv1(g, h)
        h = torch.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')  # Aggregate node features
        return hg

# GAT Model
class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_heads=8):
        super(GATModel, self).__init__()
        # num_heads is the number of attention heads
        self.conv1 = dglnn.GATConv(input_dim, hidden_dim, num_heads=num_heads)
        self.conv2 = dglnn.GATConv(hidden_dim * num_heads, hidden_dim, num_heads=num_heads)
        # Input dimension needs to be set to hidden_dim * num_heads
        self.classify = nn.Linear(hidden_dim * num_heads, num_classes)

    def forward(self, g, h, eweight=None):
        h = self.conv1(g, h)
        h = torch.relu(h)
        
        # Flatten the shape of h to meet the input requirements of conv2
        h = h.view(h.shape[0], -1)  # Flatten to (batch_size, num_heads * hidden_dim)
        
        h = self.conv2(g, h)
        g.ndata['h'] = h
        
        # Graph pooling: Aggregate node features into graph-level features
        hg = dgl.mean_nodes(g, 'h')
        
        # Flatten pooled graph-level features to fit the input of the fully connected layer
        hg = hg.view(hg.shape[0], -1)  # Flatten to (batch_size, hidden_dim * num_heads)
        
        return self.classify(hg)

# GraphSAGE Model
class GraphSAGEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GraphSAGEModel, self).__init__()
        # GraphSAGE model using mean pooling
        self.conv1 = dglnn.SAGEConv(input_dim, hidden_dim, 'mean')
        self.conv2 = dglnn.SAGEConv(hidden_dim, hidden_dim, 'mean')
        self.classify = nn.Linear(hidden_dim, num_classes)

    def forward(self, g, h, eweight=None):
        h = self.conv1(g, h)
        h = torch.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        # Graph pooling: Aggregate node features into graph-level features
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)

# TAGConv Model
class TAGConvModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(TAGConvModel, self).__init__()
        # TAGConv model
        self.conv1 = dglnn.TAGConv(input_dim, hidden_dim)
        self.conv2 = dglnn.TAGConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, num_classes)

    def forward(self, g, h, eweight=None):
        h = self.conv1(g, h)
        h = torch.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        # Graph pooling: Aggregate node features into graph-level features
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)

# DotGAT Model
class DotGATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(DotGATModel, self).__init__()
        # DotGATConv model
        self.conv1 = dglnn.DotGATConv(input_dim, hidden_dim)
        self.conv2 = dglnn.DotGATConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, num_classes)

    def forward(self, g, h, eweight=None):
        h = self.conv1(g, h)
        h = torch.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        # Graph pooling: Aggregate node features into graph-level features
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)

# Get the corresponding GNN model
def get_gnn_model(model_type, input_dim, hidden_dim, num_classes, num_heads=8):
    if model_type == 'GAT':
        return GATModel(input_dim, hidden_dim, num_classes, num_heads)
    elif model_type == 'GraphSAGE':
        return GraphSAGEModel(input_dim, hidden_dim, num_classes)
    elif model_type == 'TAGConv':
        return TAGConvModel(input_dim, hidden_dim, num_classes)
    elif model_type == 'DotGAT':
        return DotGATModel(input_dim, hidden_dim, num_classes)
    elif model_type == 'GCN':
        return GCNModel(input_dim, hidden_dim, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")