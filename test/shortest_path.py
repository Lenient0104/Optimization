import matplotlib.pyplot as plt
import networkx as nx

# Load the GraphML file
G = nx.read_graphml('map.graphml')

# Compute the positions for the nodes using a layout algorithm
pos = nx.spring_layout(G)

# Find the shortest path
shortest_path = nx.dijkstra_path(G, '389281', '389663')

# Extract the edges from the shortest path
shortest_path_edges = set(zip(shortest_path[:-1], shortest_path[1:]))

# Define the figure size
plt.figure(figsize=(10, 10))

# Draw all nodes
nx.draw_networkx_nodes(G, pos, node_color='lightblue')

# Draw all edges
nx.draw_networkx_edges(G, pos)

# Highlight the nodes in the shortest path
nx.draw_networkx_nodes(G, pos, nodelist=shortest_path, node_color='red')

# Highlight the edges in the shortest path
nx.draw_networkx_edges(G, pos, edgelist=shortest_path_edges, edge_color='red', width=2)

# Optionally add labels to the nodes
nx.draw_networkx_labels(G, pos)

# Show the plot
plt.show()
