import torch
import torch.nn as nn
import torch.optim as optim
import dgl
from dgl.nn import GNNExplainer
import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from GMCDataset import get_data_loaders
from GNNModel import get_gnn_model

# GATModel definition
class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_heads=8):
        super(GATModel, self).__init__()

        # First GATConv layer
        self.conv1 = dgl.nn.GATConv(input_dim, hidden_dim, num_heads=num_heads)

        # Second GATConv layer
        self.conv2 = dgl.nn.GATConv(hidden_dim * num_heads, hidden_dim, num_heads=num_heads)

        # Classification layer
        self.classify = nn.Linear(hidden_dim * num_heads, num_classes)

    def forward(self, graph, feat, eweight=None):
        h = self.conv1(graph, feat)
        h = torch.relu(h)

        # Flatten node features to fit the second GATConv layer
        h = h.view(h.shape[0], -1)

        h = self.conv2(graph, h)
        graph.ndata['h'] = h

        # Graph pooling to obtain graph-level features
        hg = dgl.mean_nodes(graph, 'h')
        hg = hg.view(hg.shape[0], -1)

        return self.classify(hg)

def visualize_graph_enhanced(graph, feat_mask, edge_mask, features, save_path="graph_explain_enhanced.pdf"): 
    cpu_graph = graph.cpu()
    nx_graph = cpu_graph.to_networkx()
    # Remove self-loop edges
    nx_graph.remove_edges_from(nx.selfloop_edges(nx_graph))
    # Node importance
    node_importance = (features.cpu() * feat_mask.cpu()).sum(dim=1).detach().numpy()

    # Edge importance
    edges = list(nx_graph.edges())
    edge_importance = []
    edge_mask_cpu = edge_mask.cpu().detach().numpy()
    for e in edges:
        eid = cpu_graph.edge_ids(e[0], e[1])
        edge_importance.append(edge_mask_cpu[eid])

    # Draw the graph
    plt.figure(figsize=(8,6))
    
    # Adjust layout, k can increase the distance between nodes
    pos = nx.spring_layout(nx_graph, seed=42, k=0.3)

    # Node visualization
    node_size = 1000  # Increase node size
    nx.draw_networkx_nodes(
        nx_graph, pos, node_size=node_size, node_color=node_importance, cmap=plt.cm.viridis
    )

    # Edge visualization
    nx.draw_networkx_edges(
        nx_graph, pos, edge_color=edge_importance, edge_cmap=plt.cm.coolwarm, width=2, 
        connectionstyle='arc3,rad=0.2'
    )

    # Remove node labels
    nx.draw_networkx_labels(nx_graph, pos, font_size=10, font_color="black", font_weight='bold')

    # Display colorbar for node importance
    plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), label="Node Importance")
    # plt.title("Improved Graph Visualization", fontsize=16)
    plt.axis('off')  # Turn off axes

    # Save the figure
    plt.savefig(save_path, format="pdf", dpi=800)
    plt.show()

def load_dgl(graph_path):
    # Load saved DGL graph
    graphs, _ = dgl.load_graphs(graph_path)
    return graphs[0]  # Assume only one graph in the file, return the first graph

def explain():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Initialize the model
    model = GATModel(input_dim=768, hidden_dim=128, num_classes=2)
    model = model.to(device)

    # 2. Load trained model weights
    model_path = '/home/wei/android-malware-detection-master/assets/GATExplainer'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])  # Load model weights
    model.eval()

    # 3. Load the specified DGL graph file
    graph_path = "/home/wei/GMC_family_data/kuguo/A9F26DA522FB93E399C4310854E4058CDC7AB1D43AEA13AF1FAFCB99B250B7B2_simplified.dgl"  # Replace with your actual file path
    single_graph = load_dgl(graph_path)
    single_graph = dgl.add_self_loop(single_graph)  # Add self-loops
    single_graph = single_graph.to(device)
    features = single_graph.ndata['feature'].to(device)

    # Ensure features have gradients
    features.requires_grad_()

    print("Single graph:", single_graph)
    print("Features shape:", features.shape)

    # Perform prediction
    with torch.no_grad():
        logits = model(single_graph, features)
        pred_probabilities = torch.softmax(logits, dim=1)
        pred_label = torch.argmax(pred_probabilities, dim=1).item()
        print(f"Predicted label: {pred_label}")

    # Use explainer to interpret the graph
    explainer = GNNExplainer(model, num_hops=1)
    feat_mask, edge_mask = explainer.explain_graph(single_graph, features)
    
    visualize_graph_enhanced(
        graph=single_graph,
        feat_mask=feat_mask,
        edge_mask=edge_mask,
        features=features,
        save_path="graph_explain_improved.pdf"
    )

    print("Feature importance mask:", feat_mask)
    print("Edge importance mask:", edge_mask)

if __name__ == "__main__":
    explain()