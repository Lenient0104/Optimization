import xml.etree.ElementTree as ET
from collections import defaultdict
import heapq
import matplotlib.pyplot as plt
import networkx as nx


def parse_graphml_to_adjlist(graphml_path):
    tree = ET.parse(graphml_path)
    root = tree.getroot()

    # Extract namespaces
    namespaces = {'ns': root.tag.split('}')[0].strip('{')}

    adjlist = defaultdict(dict)

    for edge in root.findall('.//ns:edge', namespaces=namespaces):
        source = edge.get('source')
        target = edge.get('target')
        weight_data = edge.find("ns:data[@key='d17']", namespaces=namespaces)
        if weight_data is not None:
            weight = float(weight_data.text)
            adjlist[source][target] = weight

    return adjlist


def convert_adjlist_to_nxgraph(adjlist):
    G = nx.DiGraph()  # Change this line to use DiGraph for a directed graph
    for node, neighbors in adjlist.items():
        for neighbor, weight in neighbors.items():
            G.add_edge(node, neighbor, weight=weight)
    return G


def display_graph(nx_graph, path=[]):
    """Visualize the graph and highlight the given path."""
    plt.figure(figsize=(12, 12))
    pos = nx.kamada_kawai_layout(nx_graph)

    # Draw the base graph
    nx.draw_networkx_nodes(nx_graph, pos, node_color="lightgray", node_size=500)
    nx.draw_networkx_edges(nx_graph, pos, width=0.5, edge_color="lightgray", arrows=True)  # Add arrows=True
    nx.draw_networkx_labels(nx_graph, pos, font_size=8)

    if path:
        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_nodes(nx_graph, pos, nodelist=path, node_color="skyblue", node_size=600)
        nx.draw_networkx_edges(nx_graph, pos, edgelist=path_edges, edge_color="blue", width=2.5,
                               arrows=True)  # Add arrows=True

    plt.show()


def dijkstra(graph, start):
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    priority_queue = [(0, start)]

    # Add predecessors for path tracking
    predecessors = {vertex: None for vertex in graph}

    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)

        # Nodes can get added to the priority queue multiple times. We only
        # process a vertex the first time we remove it from the priority queue.
        if current_distance > distances[current_vertex]:
            continue

        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight

            # Only consider this new path if it's shorter than any path we've
            # already found.
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_vertex  # Update the path for this neighbor
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances, predecessors


def get_path(predecessors, start, end):
    """Retrieve path from start to end using predecessors dictionary"""
    path = []
    while end:
        path.append(end)
        end = predecessors[end]
        if end == start:
            path.append(start)
            break
    return path[::-1]  # Reverse the path to get it from start to end


# Modify the shortest_path function
def shortest_path(graph, start, end):
    distances, predecessors = dijkstra(graph, start)
    return get_path(predecessors, start, end)


def adjlist_to_networkx(graph):
    """Convert the adjacency list to a NetworkX graph."""
    G = nx.DiGraph()  # Use DiGraph for directed graph, Graph for undirected graph
    for node, neighbors in graph.items():
        for neighbor, weight in neighbors.items():
            G.add_edge(node, neighbor, weight=weight)
    return G


# Test
graphml_path = 'your_updated_file.graphml'
graph = parse_graphml_to_adjlist(graphml_path)
nx_graph = convert_adjlist_to_nxgraph(graph)

start_node = "5105989926"
end_node = "248662261"
path = shortest_path(graph, start_node, end_node)
print("Shortest Path:", path)

display_graph(nx_graph, path)
