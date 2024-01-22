import osmnx as ox


# Replace 'path/to/your/osm_file.essential' with the path to your OSM file
osm_file = '/essential/map.graphml'

# Load the OSM file as a networkx graph
G = ox.io.load_graphml(osm_file)

# Plot the street network
ox.plot_graph(G)
