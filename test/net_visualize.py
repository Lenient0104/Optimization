import networkx as nx
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# Load the XML file
tree = ET.parse('../aco/optimization/graph202308142106.net.xml')
root = tree.getroot()

# Create a directed graph
G = nx.DiGraph()

# Iterate through the edges in the XML file and add them to the graph
for edge in root.findall(".//edge"):
    edge_id = edge.get('id')
    from_node = edge.find('lane').get('from')
    to_node = edge.find('lane').get('to')
    G.add_edge(from_node, to_node, id=edge_id)

# Draw the graph
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_size=100, node_color='skyblue', font_size=8, font_color='black', font_weight='bold')
edge_labels = {edge: data['id'] for edge, data in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

plt.title("NetworkX Visualization of graph.net.xml")
plt.axis('off')
plt.show()
