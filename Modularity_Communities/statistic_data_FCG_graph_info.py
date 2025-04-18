import os
import numpy as np
from pathlib import Path
import dgl

def calculate_graph_statistics(dgl_dirs):
    # Store node counts, edge counts, and degree information
    node_counts = []
    edge_counts = []
    degrees = []  # Store degrees of all nodes across all graphs
    total_size = 0  # Total disk usage (bytes)
    total_graphs = 0  # Total number of graphs
    
    # Iterate through all malware family directories
    for dgl_dir in dgl_dirs:
        family_folder = Path(dgl_dir)
        print(f"Processing family folder: {family_folder}")

        # Get all .dgl files in the current directory
        dgl_files = [f for f in family_folder.iterdir() if f.suffix == '.dgl']

        # Iterate through each DGL file
        for dgl_file in dgl_files:
            try:
                # Load DGL graph
                graphs, _ = dgl.data.utils.load_graphs(str(dgl_file))
                total_graphs += len(graphs)

                # Collect node and edge counts for each graph
                for g in graphs:
                    node_counts.append(g.number_of_nodes())
                    edge_counts.append(g.number_of_edges())
                    
                    # Collect degree information for all nodes
                    degrees.extend(g.in_degrees().tolist())  # Get in-degree of each node
                    
                # Calculate disk usage of the file (bytes)
                total_size += dgl_file.stat().st_size
                
            except Exception as e:
                print(f"Error processing {dgl_file}: {e}")
                continue
    
    # If no valid graphs are found, print a warning and exit
    if not node_counts or not edge_counts or not degrees:
        print(f"No valid graphs found in {dgl_dirs}.")
        return
    
    # Convert to NumPy arrays for computation
    node_counts = np.array(node_counts)
    edge_counts = np.array(edge_counts)
    degrees = np.array(degrees)

    # Calculate node count statistics
    node_avg = np.mean(node_counts)
    node_median = np.median(node_counts)
    node_max = np.max(node_counts)
    node_min = np.min(node_counts)

    # Calculate edge count statistics
    edge_avg = np.mean(edge_counts)
    edge_median = np.median(edge_counts)
    edge_max = np.max(edge_counts)
    edge_min = np.min(edge_counts)

    # Calculate degree statistics
    degree_avg = np.mean(degrees)
    degree_median = np.median(degrees)
    degree_max = np.max(degrees)
    degree_min = np.min(degrees)

    # Calculate disk usage (in GB)
    total_size_gb = total_size / (1024 ** 3)

    # Print statistics
    print(f"Graph Data Statistics:")
    print(f"Total number of graphs: {total_graphs}")
    
    print(f"\nNode Count Statistics:")
    print(f"Average node count: {node_avg}")
    print(f"Median node count: {node_median}")
    print(f"Maximum node count: {node_max}")
    print(f"Minimum node count: {node_min}")
    
    print(f"\nEdge Count Statistics:")
    print(f"Average edge count: {edge_avg}")
    print(f"Median edge count: {edge_median}")
    print(f"Maximum edge count: {edge_max}")
    print(f"Minimum edge count: {edge_min}")

    print(f"\nDegree Statistics:")
    print(f"Average degree: {degree_avg}")
    print(f"Median degree: {degree_median}")
    print(f"Maximum degree: {degree_max}")
    print(f"Minimum degree: {degree_min}")

    print(f"\nDisk Usage: {total_size_gb:.2f} GB")

# Example: Specify the parent directory paths containing malware family folders
dgl_dirs = [
    '/home/wei/GMC_family_data/adpush/',
    '/home/wei/GMC_family_data/artemis/',
    '/home/wei/GMC_family_data/dzhtny/',
    '/home/wei/GMC_family_data/igexin/',
    '/home/wei/GMC_family_data/kuguo/',
    '/home/wei/GMC_family_data/leadbolt/',
    '/home/wei/GMC_family_data/openconnection/',
    '/home/wei/GMC_family_data/spyagent/'
]

benign_BRG_dir = ["/home/wei/android-malware-detection-master/data/benign"]

calculate_graph_statistics(benign_BRG_dir)