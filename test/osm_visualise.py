import osmnx as ox

# Load the OSM file as a graph
graph = ox.graph_from_xml('map (5).osm')

# Save the graph in GraphML format
filepath = "big-2.graphml"
ox.io.save_graphml(graph, filepath)

# Load the GraphML file as a networkx graph
G = ox.io.load_graphml('big-2.graphml')

# Plot the street network
ox.plot_graph(G)
