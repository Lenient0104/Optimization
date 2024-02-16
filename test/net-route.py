import xml.etree.ElementTree as ET
import csv
import networkx as nx
import matplotlib.pyplot as plt


def get_edge_weights(csv_file):
    edge_weights = {}
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # skip header
        for row in reader:
            edge_id = row[2]
            mean_speed = float(row[3])
            edge_weights[edge_id] = 1 / mean_speed  # We'll use inverse of speed as weight (time taken)
    return edge_weights


def create_graph_from_xml(net_xml_file, edge_weights):
    tree = ET.parse(net_xml_file)
    root = tree.getroot()

    G = nx.DiGraph()

    for edge in root.findall('edge'):
        from_node = edge.get('from')
        to_node = edge.get('to')
        edge_id = edge.get('id')
        if from_node and to_node:
            G.add_edge(from_node, to_node, weight=edge_weights.get(edge_id, float('inf')))

    return G


def dijkstra(graph, start, end):
    shortest_distances = {node: float('infinity') for node in graph.nodes()}
    shortest_distances[start] = 0
    shortest_path = {node: [] for node in graph.nodes()}
    shortest_path[start] = [start]
    unvisited_nodes = set(graph.nodes())

    current_node = start

    while current_node != end:
        unvisited_nodes.remove(current_node)
        for neighbor, attributes in graph[current_node].items():
            distance_to_neighbour = attributes['weight']
            new_distance = shortest_distances[current_node] + distance_to_neighbour

            if new_distance < shortest_distances[neighbor]:
                shortest_distances[neighbor] = new_distance
                shortest_path[neighbor] = shortest_path[current_node] + [neighbor]

        next_nodes = {node: shortest_distances[node] for node in unvisited_nodes}
        if not next_nodes:
            return "Path not reachable", float('inf')
        current_node = min(next_nodes, key=next_nodes.get)

    return shortest_path[end], shortest_distances[end]


def compute_shortest_path_and_visualize(net_xml_file, csv_file, start_node, end_node):
    edge_weights = get_edge_weights(csv_file)
    G = create_graph_from_xml(net_xml_file, edge_weights)
    path, distance = dijkstra(G, start_node, end_node)

    pos = nx.spring_layout(G)
    plt.figure(figsize=(15, 15))

    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)

    if path != "Path not reachable":
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r', width=2)

    plt.show()

    return path, distance


# Example usage:
net_xml_file = '../aco/optimization/graph202308142106.net.xml'
csv_file = '../aco/optimization/graph202308142106_edges.csv'
start_node = '8396907063'
end_node = '530941809'
path, distance = compute_shortest_path_and_visualize(net_xml_file, csv_file, start_node, end_node)
print(f"Shortest path from {start_node} to {end_node}: {path} with distance {distance}")
