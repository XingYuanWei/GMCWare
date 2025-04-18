import random
from torch.utils.data import Dataset, DataLoader, random_split
import dgl
import torch
from pathlib import Path


# Directories for benign and malware data
BENIGN_DIR = "/home/wei/android-malware-detection-master/data/benign"
MALWARE_DIRS = [
    "/home/wei/GMC_family_data/adpush",
    "/home/wei/GMC_family_data/artemis",
    "/home/wei/GMC_family_data/dzhtny",
    "/home/wei/GMC_family_data/igexin",
    "/home/wei/GMC_family_data/kuguo",
    "/home/wei/GMC_family_data/leadbolt",
    "/home/wei/GMC_family_data/openconnection",
    "/home/wei/GMC_family_data/spyagent",
]

# Number of samples to extract from each malware folder
MALWARE_SAMPLES_PER_CLASS = 160  # 1280 malware samples / 8 classes = 160 per class


def load_graphs_from_directories(directories: list, label: int, max_samples_per_class=None):
    graphs = []
    labels = []
    graph_paths = []

    for directory in directories:
        dir_path = Path(directory)
        all_graph_files = [graph_file for graph_file in dir_path.iterdir() if graph_file.suffix == ".dgl"]

        # If a maximum number of samples per class is specified, randomly sample
        if max_samples_per_class:
            all_graph_files = random.sample(all_graph_files, min(max_samples_per_class, len(all_graph_files)))

        for graph_file in all_graph_files:
            try:
                graph, _ = dgl.load_graphs(str(graph_file))
                graphs.append(graph[0])
                labels.append(label)
                graph_paths.append(str(graph_file))
            except Exception as e:
                print(f"Error loading {graph_file}: {e}")
    return graphs, torch.tensor(labels), graph_paths


def load_data():
    """
    Load benign and malware data.
    :return: Combined list of graphs, labels, and paths
    """
    # Load benign software data
    benign_graphs, benign_labels, benign_paths = load_graphs_from_directories([BENIGN_DIR], label=0)

    # Load malware data, randomly sampling a specified number of samples from each folder
    malware_graphs, malware_labels, malware_paths = load_graphs_from_directories(
        MALWARE_DIRS, label=1, max_samples_per_class=None
    )

    graphs = benign_graphs + malware_graphs
    labels = torch.cat([benign_labels, malware_labels], dim=0)
    paths = benign_paths + malware_paths

    print("Benign count =", len(benign_graphs))
    print("Malware count =", len(malware_graphs))
    # print(malware_paths)
    return graphs, labels, paths


class GraphDataset(Dataset):
    def __init__(self, graphs, labels, graph_paths):
        self.graphs = graphs
        self.labels = labels
        self.graph_paths = graph_paths

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx], self.graph_paths[idx]


def collate_fn(batch):
    graphs, labels, paths = zip(*batch)
    graphs = [add_self_loops_if_needed(graph) for graph in graphs]
    batched_graph = dgl.batch(graphs)
    labels = torch.stack(labels)
    return batched_graph, labels, paths


def add_self_loops_if_needed(graph):
    if graph.in_degrees().min() == 0:
        graph = dgl.add_self_loop(graph)
    return graph


def split_dataset(dataset, train_ratio=0.8):
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])


def get_data_loaders(batch_size=4, train_ratio=0.8):
    """
    Interface: Load benign and malware data, and return training and testing data loaders
    :param batch_size: Size of each batch
    :param train_ratio: Proportion of the training set
    :return: train_loader and test_loader
    """
    # Load data
    graphs, labels, graph_paths = load_data()

    # Construct dataset
    dataset = GraphDataset(graphs, labels, graph_paths)

    # Split dataset
    train_dataset, test_dataset = split_dataset(dataset, train_ratio)

    # Construct data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, test_loader


if __name__ == "__main__":
    # Get training and testing data loaders
    train_loader, test_loader = get_data_loaders(batch_size=4, train_ratio=0.8)
    print(len(train_loader))
    print(len(test_loader))
    # Example: Iterate through training data
    # for batched_graph, labels, paths in train_loader:
    #     print("Batched graph:", batched_graph)
    #     print("Labels:", labels)
    #     print("Paths:", paths)