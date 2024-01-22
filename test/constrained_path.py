import networkx as nx
import matplotlib.pyplot as plt
import csv
import xml.etree.ElementTree as ET


def parse_net_xml(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    edges = {}

    for edge in root.findall('edge'):
        edge_id = edge.attrib['id']
        if 'from' in edge.attrib and 'to' in edge.attrib:
            for lane in edge.findall('lane'):
                lane_id = lane.attrib['id']
                edges[edge_id] = {
                    'from': edge.attrib['from'],
                    'to': edge.attrib['to'],
                    'length': float(lane.attrib['length']),
                    'speed_limit': float(lane.attrib['speed']),
                    'shape': lane.attrib['shape']
                }
    return edges


def map_speeds_to_lanes(filename, edges):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            lane_id = row[2]
            mean_speed = float(row[3])
            if lane_id in edges:
                edges[lane_id]['speed'] = mean_speed
                edges[lane_id]['time'] = edges[lane_id]['length'] / mean_speed if mean_speed != 0 else float('inf')


def build_graph_from_edges(edges):
    G = nx.DiGraph()
    for lane_id, data in edges.items():
        G.add_edge(data['from'], data['to'],
                   weight=data.get('time', float('inf')),
                   lane_id=lane_id,
                   length=data['length'],
                   speed=data.get('speed', 0))
    return G


def dijkstra_with_max_time(G, source, target, max_time):
    dist = {node: float('infinity') for node in G}
    dist[source] = 0
    pred = {node: None for node in G}
    to_explore = [(0, source)]

    while to_explore:
        d, current = min(to_explore)
        to_explore.remove((d, current))
        if current == target and d <= max_time:
            path = []
            while current:
                path.append(current)
                current = pred[current]
            return path[::-1]
        for neighbor in G[current]:
            new_dist = d + G[current][neighbor]['weight']
            if new_dist < dist[neighbor] and new_dist <= max_time:
                dist[neighbor] = new_dist
                pred[neighbor] = current
                to_explore.append((new_dist, neighbor))
    return None


def constrained_shortest_path(G, source, target, max_time):
    path = dijkstra_with_max_time(G, source, target, max_time)
    if not path:
        path = nx.shortest_path(G, source, target, weight='weight')
        total_time = sum(G[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
        print(f"Warning: The shortest path takes {total_time:.2f} which exceeds the limit of {max_time}.")
        return path
    total_time = sum(G[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
    print(f"The shortest path within the limit takes {total_time:.2f}.")
    return path


def dijkstra_top_three_routes(G, source, target, max_time):
    dist = {node: float('infinity') for node in G}
    dist[source] = 0
    pred = {node: None for node in G}
    to_explore = [(0, source)]
    found_paths = []

    while to_explore and len(found_paths) < 3:
        d, current = min(to_explore)
        to_explore.remove((d, current))

        # If the current node is the target and the distance is within max_time
        if current == target and d <= max_time:
            path = []
            while current:
                path.append(current)
                current = pred[current]
            found_paths.append((d, path[::-1]))
            continue  # Continue looking for other paths after adding this one

        for neighbor in G[current]:
            new_dist = d + G[current][neighbor]['weight']
            if new_dist < dist[neighbor] and new_dist <= max_time:
                dist[neighbor] = new_dist
                pred[neighbor] = current
                to_explore.append((new_dist, neighbor))

    # Sort the found paths by their distance and return them
    found_paths.sort(key=lambda x: x[0])
    return [route for _, route in found_paths]


def plot_graph_with_path(G, path):
    plt.figure(figsize=(50, 50))
    pos = nx.spring_layout(G)  # Define the layout for our nodes
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edge_color='grey')

    if path:
        edges_in_path = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=edges_in_path, edge_color='r', width=2)
    plt.show()


# Parsing .net.xml
edges = parse_net_xml('../optimization/graph202308142106.net.xml')
map_speeds_to_lanes('../optimization/graph202308142106_edges.csv', edges)
G = build_graph_from_edges(edges)

# Define your constraints and source, target nodes
source_node = '5355515613'
target_node = '389281'
max_time_limit = 500

path = constrained_shortest_path(G, source_node, target_node, max_time_limit)

# Visualizing the graph and the route
plot_graph_with_path(G, path)

print(dijkstra_top_three_routes(G, source_node, target_node, max_time_limit))
