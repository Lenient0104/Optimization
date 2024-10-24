import networkx as nx
import pandas as pd
import csv
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET


class Optimization:
    def __init__(self, net_xml_path, edges_csv_path):
        # Parse the XML file to get road and lane information
        self.edges = self.parse_net_xml(net_xml_path)
        # Map speeds to lanes using the provided CSV file
        self.map_speeds_to_lanes(edges_csv_path, self.edges)
        # Build a graph based on the edge data
        self.G = self.build_graph_from_edges(self.edges)

    def parse_net_xml(self, filepath):
        # Parse the XML file and get the root element
        tree = ET.parse(filepath)
        root = tree.getroot()
        edges = {}

        # Iterate over the 'edge' elements in the XML
        for edge in root.findall('edge'):
            edge_id = edge.attrib['id']
            # Check if both source and target nodes are provided for the edge
            if 'from' in edge.attrib and 'to' in edge.attrib:
                # Iterate over lanes inside each edge
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

    def map_speeds_to_lanes(self, filename, edges):
        # Load the CSV file into a DataFrame
        df = pd.read_csv(filename)

        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            # Get lane_id and mean_speed for each row
            lane_id = row['edge_id']
            mean_speed = row['mean_speed']

            # If lane exists in edges, map the speed and compute time taken
            if lane_id in edges:
                edges[lane_id]['speed'] = mean_speed
                edges[lane_id]['time'] = edges[lane_id]['length'] / mean_speed if mean_speed != 0 else float('inf')

    def build_graph_from_edges(self, edges):
        # Initialize a directed graph
        G = nx.DiGraph()

        # Add edges to the graph using the data provided
        for lane_id, data in edges.items():
            G.add_edge(data['from'], data['to'],
                       weight=data.get('time', float('inf')),
                       lane_id=lane_id,
                       length=data['length'],
                       speed=data.get('speed', 0))
        return G

    def constrained_shortest_path(self, source, target, max_time):
        # Find the shortest path with a constraint on maximum time
        path = self.dijkstra_with_max_time(self.G, source, target, max_time)
        if not path:
            # If no path found within the constraint, find the absolute shortest path
            path = nx.shortest_path(self.G, source, target, weight='weight')
            total_time = sum(self.G[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
            print(f"Warning: The shortest path takes {total_time:.2f} which exceeds the limit of {max_time}.")
            return path
        total_time = sum(self.G[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
        print(f"The shortest path within the limit takes {total_time:.2f}.")
        return path

    def dijkstra_with_max_time(self, G, source, target, max_time):
        # Dijkstra's algorithm modified with a maximum time constraint
        dist = {node: float('infinity') for node in G}
        dist[source] = 0
        pred = {node: None for node in G}
        to_explore = [(0, source)]

        # Main loop for Dijkstra's algorithm
        while to_explore:
            d, current = min(to_explore)
            to_explore.remove((d, current))
            if current == target and d <= max_time:
                # Construct the path from source to target
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

    def dijkstra_top_three_routes(self, source, target, max_time, G):
        # Find top 3 shortest paths using Dijkstra's algorithm with a maximum time constraint
        dist = {node: float('infinity') for node in G}
        dist[source] = 0
        pred = {node: None for node in G}
        to_explore = [(0, source)]
        found_paths = []

        while to_explore and len(found_paths) < 3:
            d, current = min(to_explore)
            to_explore.remove((d, current))

            if current == target and d <= max_time:
                path = []
                while current:
                    path.append(current)
                    current = pred[current]
                found_paths.append((d, path[::-1]))
                continue

            for neighbor in G[current]:
                new_dist = d + G[current][neighbor]['weight']
                if new_dist < dist[neighbor] and new_dist <= max_time:
                    dist[neighbor] = new_dist
                    pred[neighbor] = current
                    to_explore.append((new_dist, neighbor))

        found_paths.sort(key=lambda x: x[0])
        return [route for _, route in found_paths]

    def plot_graph_with_path(self, path, G):
        # Plot the graph with a particular path highlighted
        plt.figure(figsize=(50, 50))
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, edge_color='grey')

        if path:
            edges_in_path = list(zip(path[:-1], path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=edges_in_path, edge_color='r', width=2)
        plt.show()

    def analyze_route(self, source_node, target_node, max_time_limit):
        # Analyze the best route from source to target with a maximum time constraint
        path = self.constrained_shortest_path(source_node, target_node, max_time_limit)
        self.plot_graph_with_path(path)
        return self.dijkstra_top_three_routes(source_node, target_node, max_time_limit)
