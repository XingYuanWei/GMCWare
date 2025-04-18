import os
from typing import Dict
from pathlib import Path
from collections import defaultdict
import sys
import traceback
import argparse
import multiprocessing as mp
import torch.multiprocessing as mp_torch

from androguard.misc import AnalyzeAPK
from androguard.core.analysis.analysis import ExternalMethod
from androguard.core.dex import EncodedMethod, DalvikCode

import torch
import networkx as nx
import dgl

import matplotlib.pyplot as plt
from networkx.algorithms.community import greedy_modularity_communities

from transformers import BertTokenizer, BertModel


# Instruction category mapping
INSTRUCTION_CATEGORIES = {
    'arithmetic_operations': ['add', 'sub', 'mul', 'div', 'mod'],
    'memory_operations': ['load', 'store', 'move'],
    'control_flow_operations': ['jmp', 'branch', 'call'],
    'external_calls': ['invoke']
}

class BertVectorizer:
    def __init__(self):
        # Use local pretrained model
        self.tokenizer = BertTokenizer.from_pretrained("/home/wei/android-malware-detection-master/bert")
        self.model = BertModel.from_pretrained("/home/wei/android-malware-detection-master/bert")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def vectorize(self, text: str) -> torch.Tensor:
        """
        Vectorize text using BERT
        :param text: Input description text
        :return: Dense vector (768 dimensions)
        """
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
        tokens = {key: val.to(self.device) for key, val in tokens.items()}
        with torch.no_grad():
            output = self.model(**tokens)
        return output.last_hidden_state.mean(dim=1).squeeze()  # Average pooling to get sentence vector



def extract_opcode_and_bytecode(function) -> str:
    """ Extract opcode and bytecode of a function as text description

    :param function: ExternalMethod or EncodedMethod object
    :return: String description
    """
    description = []
    instructions = []
    try:
        # If it is an ExternalMethod, extract class name, method name, etc.
        if isinstance(function, ExternalMethod):
            description.append(f"ExternalMethod: {function.class_name}->{function.name}{function.descriptor}")
        
        # If it is an EncodedMethod, extract bytecode and opcode
        elif isinstance(function, EncodedMethod):
            function.load()
            code = function.get_code()      
            # If the method has bytecode (DalvikCode or DCode object)
            if isinstance(code, DalvikCode):
                instructions = code.get_bc().get_instructions()  # Get bytecode instructions
                for idx, instruction in enumerate(instructions):
                    bytecode = instruction.get_raw()
                    op_name = instruction.get_name()
                    bytecode_hex = " ".join([f"{byte:02x}" for byte in bytecode])
                    description.append(f"{op_name} {bytecode_hex}")
            else:
                description.append("Unknown code type")

        else:
            description.append(f"Unknown method type: {type(function)}")
    
    except Exception as e:
        print(f"Error extracting opcode/bytecode: {e}")
    
    return description, list(instructions)  # Return instructions and description


def cluster_instructions(instructions):
    """
    Cluster instructions, grouping them into different categories
    :param instructions: List of instructions
    :return: Frequency of each category
    """
    # Use a regular dictionary instead of defaultdict
    clustered_instructions = {
        'arithmetic_operations': 0,
        'memory_operations': 0,
        'control_flow_operations': 0,
        'external_calls': 0
    }

    for instruction in instructions:
        op_name = instruction.get_name()
        for category, ops in INSTRUCTION_CATEGORIES.items():
            if op_name in ops:
                clustered_instructions[category] += 1
                break

    return clustered_instructions



def compute_community_features(community_mapping: Dict, original_node_mapping: Dict, vectorizer: BertVectorizer) -> Dict:
    """
    Aggregate community node features, use BERT to vectorize community bytecode and opcode descriptions, and compute instruction clustering features
    :param community_mapping: Mapping of nodes to communities
    :param original_node_mapping: Mapping of string node identifiers to original nodes
    :param vectorizer: BERT vectorizer
    :return: Community feature vectors and instruction clustering features
    """
    community_descriptions = {community: [] for community in set(community_mapping.values())}
    community_instruction_features = {community: defaultdict(int) for community in set(community_mapping.values())}

    for node, community in community_mapping.items():
        original_node = original_node_mapping[node]  # Get original node object
        # Extract node feature description
        description, instructions = extract_opcode_and_bytecode(original_node)
        # Cluster instructions and compute frequency of each category
        instruction_features = cluster_instructions(instructions)  # Avoid large amounts of hexadecimal data interference

        # Combine instruction clustering features with other static analysis features
        community_descriptions[community].append(description) # Aggregate node text descriptions
        community_descriptions[community].append(str(instruction_features))  # Ensure it is a string

        # Aggregate instruction features (accumulate frequency) 
        for key, value in instruction_features.items():
            community_instruction_features[community][key] += value

    # Vectorize descriptions for each community
    community_features = {}
    for community, descriptions in community_descriptions.items():
        # Ensure each element in descriptions is a string
        descriptions = [str(desc) for desc in descriptions]  # Convert to strings
        combined_description = " ".join(descriptions)  # Join descriptions into a single string
        if combined_description.strip():  # Ensure description is not empty
            community_features[community] = vectorizer.vectorize(combined_description)
        else:
            community_features[community] = torch.zeros(768)  # Default zero vector
    
     # Ensure clustering features are also torch.Tensor
    for community, instruction_features in community_instruction_features.items():
        community_instruction_features[community] = torch.tensor(
            [instruction_features[key] for key in INSTRUCTION_CATEGORIES.keys()], dtype=torch.float32
        )

    return community_features, community_instruction_features


def simplify_and_compute_features(graph: nx.DiGraph, vectorizer: BertVectorizer) -> nx.DiGraph:
    """
    Simplify the call graph using modularity partitioning and compute community feature vectors
    :param graph: Original call graph
    :param vectorizer: BERT vectorizer
    :return: Simplified graph with community features
    """
    try:
        # Create node identifier mapping while preserving original nodes
        node_mapping = {node: f"Node_{i}" for i, node in enumerate(graph.nodes())}
        original_node_mapping = {f"Node_{i}": node for i, node in enumerate(graph.nodes())}

        # Create a new graph with simple identifiers as nodes
        simplified_graph = nx.DiGraph()
        for node in graph.nodes():
            simplified_graph.add_node(node_mapping[node])

        # Create edges between nodes
        for u, v in graph.edges():
            simplified_graph.add_edge(node_mapping[u], node_mapping[v])

        # Obtain community partitioning
        communities = list(greedy_modularity_communities(simplified_graph))
        community_mapping = {}
        for i, community in enumerate(communities):
            community_node = f"Community_{i}"
            for node in community:
                community_mapping[node] = community_node

        # Merge connections between communities
        final_graph = nx.DiGraph()
        for u, v in simplified_graph.edges():
            u_community = community_mapping.get(u)
            v_community = community_mapping.get(v)
            if u_community != v_community:
                final_graph.add_edge(u_community, v_community)

        # Compute community features and instruction clustering features
        community_features, community_instruction_features = compute_community_features(community_mapping, original_node_mapping, vectorizer)


        # Add features to nodes
        nx.set_node_attributes(final_graph, community_features, "feature")
        nx.set_node_attributes(final_graph, community_instruction_features, "instruction_features")
        return final_graph

    except Exception as e:
        print(f"Error simplifying graph: {e}")
        return graph  # Return original graph if simplification fails


def plot_graph(graph: nx.DiGraph):
    """
    Visualize the simplified call graph
    :param graph: NetworkX graph
    """
    try:
        # Compute graph layout
        pos = nx.spring_layout(graph, seed=42, k=30)  # Use spring layout, k increases distance between nodes
        node_color = [graph.degree(node) for node in graph.nodes()]  # Node color varies based on degree

        # Draw the graph
        plt.figure(figsize=(12, 12))
        nx.draw(
            graph, pos, with_labels=True, node_size=1000, node_color=node_color, cmap=plt.cm.Blues,
            font_size=5, font_weight='bold', edge_color='gray'
        )
        plt.title("Simplified Function Call Graph")
        plt.show()
        plt.savefig('pic.png',dpi=500)
    except Exception as e:
        print(f"Error while plotting the graph: {e}")


def process_apk(source_file, dest_dir, vectorizer):
    """
    Process a single APK file, extract the simplified function call graph, and compute community features
    :param source_file: APK file path
    :param dest_dir: Target directory to save simplified graphs
    :param vectorizer: BERT vectorizer
    """
    try:
        file_name = source_file.stem
        dest_file = dest_dir / f'{file_name}_simplified.dgl'

        # Check if target file already exists; skip processing if it does
        if dest_file.exists():
            print(f"{file_name} already processed, skipping.")
            return
        
        
        _, _, dx = AnalyzeAPK(source_file)
        cg = dx.get_call_graph()

        # Convert to NetworkX graph
        nx_cg = nx.DiGraph()
        for node in cg.nodes():
            nx_cg.add_node(node)
        for u, v in cg.edges():
            nx_cg.add_edge(u, v)

        # Simplify the call graph and compute community features
        simplified_cg = simplify_and_compute_features(nx_cg, vectorizer)
        # Plot the graph
        # plot_graph(simplified_cg)
        print(simplified_cg)
        print(cg)

        # Convert to DGL graph
        dg = dgl.from_networkx(simplified_cg, node_attrs=["feature","instruction_features"])
        dest_file = dest_dir / f'{file_name}_simplified.dgl'
        dgl.data.utils.save_graphs(str(dest_file), [dg])
        print(f"Processed and simplified {source_file}")
    except Exception:
        print(f"Error while processing {source_file}")
        traceback.print_exception(*sys.exc_info())


def test():
    vectorizer = BertVectorizer()
    apk_path = "/storage/xiaowei_data/purity_data/2015/gap10/malware_20_29/E9C85EEB93B9866276DB66E6F3EB64C76C19BFB3D91F938D44B0CC9C7801302E.apk"
    dest_dir = '/home/wei/android-malware-detection-master/data/'
    process_apk(Path(apk_path),Path(dest_dir),vectorizer)


if __name__ == '__main__':
    # test()

    parser = argparse.ArgumentParser(description='Batch Process APKs into Simplified Function Call Graphs with BERT Features')
    # parser.add_argument(
    #     '-s', '--source-dir',
    #     help='The directory containing APK files',
    #     required=True
    # )
    # parser.add_argument(
    #     '-d', '--dest-dir',
    #     help='The directory to store simplified graphs',
    #     required=True
    # )
    # parser.add_argument(
    #     '--n-jobs',
    #     default=multiprocessing.cpu_count(),
    #     help='Number of jobs to use for processing'
    # )
    # parser.add_argument(
    #     '--limit',
    #     help='Limit the number of APKs to process',
    #     default=-1
    # )
      # Set multiprocessing start method to 'spawn'
    mp.set_start_method('spawn', force=True)

    # Configure parameters
    args = parser.parse_args()
    args.source_dir = "/home/wei/android-malware-detection-master/obfu_malware_20_29_APK/RandomManifest"
    args.dest_dir = '/home/wei/android-malware-detection-master/data/obfu_malware_20_29/RandomManifest'
    args.limit = 0
    args.n_jobs = 60

    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f'{source_dir} not found')
    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    vectorizer = BertVectorizer()

    # Get APK files to process
    files = [x for x in source_dir.iterdir() if x.suffix == '.apk']
    print(len(files))
    limit = int(args.limit)
    # if limit != -1:
    #     files = files[:limit]

    print(f"Starting to process {len(files)} APK files with {args.n_jobs} jobs...")
    # Use torch.multiprocessing Pool
    J = mp_torch.Pool(args.n_jobs)
    J.starmap(process_apk, [(file, dest_dir, vectorizer) for file in files])
    J.close()