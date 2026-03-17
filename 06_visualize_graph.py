import networkx as nx
import pickle
import matplotlib.pyplot as plt
from config_01 import AUGMENTED_GRAPH_PATH

def visualize_subgraph(target_node="pneumonia"):
    print(f"Loading graph from {AUGMENTED_GRAPH_PATH}...")
    try:
        with open(AUGMENTED_GRAPH_PATH, 'rb') as f:
            G = pickle.load(f)
    except FileNotFoundError:
        print("Error: Graph file not found. Please run previous scripts first.")
        return

    # Check if node exists (handling case sensitivity)
    if target_node not in G:
        print(f"Node '{target_node}' not found in graph. Trying case-insensitive search...")
        found = False
        for n in G.nodes():
            if str(n).lower() == target_node.lower():
                target_node = n
                found = True
                break
        if not found:
            print(f"Cannot find node '{target_node}'.")
            return

    print(f"Visualizing subgraph for: {target_node}")

    # Extract neighbor nodes (1-hop)
    neighbors = list(G.neighbors(target_node)) + list(G.predecessors(target_node))
    subgraph_nodes = neighbors + [target_node]
    subG = G.subgraph(subgraph_nodes)

    # Layout configuration
    pos = nx.spring_layout(subG, seed=42, k=0.5)
    
    plt.figure(figsize=(12, 8))
    
    # Define colors based on node type
    colors = []
    labels = {}
    
    for node in subG.nodes():
        node_type = subG.nodes[node].get('type', 'unknown')
        labels[node] = node  # Label is the node name
        
        if node == target_node:
            colors.append('red')      # Center Disease
        elif node_type == 'symptom':
            colors.append('#90EE90')  # Light Green
        elif node_type == 'diagnostic_difference':
            colors.append('#FFD700')  # Gold (LLM Knowledge)
        elif node_type == 'subcategory':
            colors.append('#87CEEB')  # Sky Blue
        elif node_type == 'system':
            colors.append('#00008B')  # Dark Blue
        else:
            colors.append('gray')

    # Draw nodes
    nx.draw_networkx_nodes(subG, pos, node_color=colors, node_size=1500, alpha=0.9)
    
    # Draw edges
    nx.draw_networkx_edges(subG, pos, width=1.0, alpha=0.5, arrowsize=20)
    
    # Draw labels
    nx.draw_networkx_labels(subG, pos, labels, font_size=8, font_weight="bold")

    # Add Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Target Disease'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#87CEEB', label='Category'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#90EE90', label='Symptom'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFD700', label='LLM Difference'),
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.title(f"Knowledge Graph Fragment: {target_node}", fontsize=16)
    plt.axis('off')
    
    output_file = "kg_visualization.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_file}. Open it to see your graph!")
    plt.show()

if __name__ == "__main__":
    # You can change 'pneumonia' to any other disease ID from your data
    visualize_subgraph("Pneumonia")