import dgl
import matplotlib.pyplot as plt
import networkx as nx

ATTRIBUTES = ['external', 'entrypoint', 'native', 'public', 'static', 'codesize', 'api', 'user']

def load_and_plot_dgl_graph(graph_path: str, output_svg_path: str, max_nodes=None):
    # Load graph from disk
    graphs, _ = dgl.data.utils.load_graphs(graph_path)
    dg = graphs[0]  # Assume only one graph in the file

    # Convert DGL graph to NetworkX graph
    nx_graph = dg.to_networkx(node_attrs=ATTRIBUTES)

    # Optional: Reduce the number of nodes
    if max_nodes:
        nodes_to_remove = list(nx_graph.nodes())[max_nodes:]
        nx_graph.remove_nodes_from(nodes_to_remove)

    # Use spring_layout to spread out nodes
    pos = nx.spring_layout(nx_graph, k=0.5, iterations=50)  # Adjust k and iterations to control spread

    # Draw the graph, omitting node labels and reducing node size
    plt.figure(figsize=(12, 12))
    nx.draw(nx_graph, pos, with_labels=False, node_size=20, node_color='skyblue', edge_color='gray', width=0.5)

    # Save as SVG format
    plt.savefig(output_svg_path, format='svg')
    print(f"Graph saved as SVG at {output_svg_path}")
    plt.close()  # Close the figure to release resources


# Load and plot the graph
graph_path = '/home/wei/GNNData_Traditional/2018/malware_20_29/29A29917710DADB15324586724C3DEBB0E7F665225308636E711785E8A68E868.fcg'
output_pdf_path = 'output_graph.pdf' 
load_and_plot_dgl_graph(graph_path, output_pdf_path)