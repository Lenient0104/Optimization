import time
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from tqdm import tqdm


class Optimization:
    def __init__(self, net_xml_path, edges_csv_path):
        self.net_xml_path = net_xml_path
        self.edges_csv_path = edges_csv_path
        self.edge_mapping = self.extract_edge_info()
        # Parse the XML file to get road and lane information
        self.edges = self.parse_net_xml(net_xml_path)
        # Map speeds to lanes using the provided CSV file
        self.map_speeds_to_lanes(edges_csv_path, self.edges)
        # Build a graph based on the edge data
        self.G = self.build_graph_from_edges(self.edges)

    def extract_edge_info(self):
        tree = ET.parse(self.net_xml_path)
        root = tree.getroot()
        edge_mapping = {}
        for edge in root.findall(".//edge"):
            edge_id = edge.get('id')
            from_node = edge.get('from')
            to_node = edge.get('to')
            edge_mapping[(from_node, to_node)] = edge_id
            edge_mapping[(to_node, from_node)] = "-" + edge_id
        return edge_mapping

    def convert_node_path_to_edge_path(self, node_path):
        edge_path = []
        for i in range(len(node_path) - 1):
            edge_id = self.edge_mapping.get((node_path[i], node_path[i + 1]))
            if edge_id is None:
                raise ValueError(f"No edge found for nodes {node_path[i]} -> {node_path[i + 1]}")
            edge_path.append(edge_id)
        return edge_path

    def parse_net_xml(self, filepath):
        tree = ET.parse(filepath)
        root = tree.getroot()
        edges = {}

        # Creating a map for edge details from <edge> tags
        edge_detail_map = {}
        start_time = time.time()
        for edge in tqdm(root.findall('edge'), desc="Processing edges"):
            for lane in edge.findall('lane'):
                edge_id = edge.attrib['id']
                edge_detail_map[edge_id] = {
                    'length': float(lane.attrib['length']),
                    'speed_limit': float(lane.attrib['speed']),
                    'shape': lane.attrib['shape']
                }
        end_time = time.time()  # End timing
        print(f"Finished information extraction from edges in {end_time - start_time:.2f} seconds.")


        start_time = time.time()
        # Constructing edges from <connection> tags
        for conn in tqdm(root.findall('connection'), desc="Processing connections"):
            from_edge = conn.get('from')
            to_edge = conn.get('to')

            # Forming a unique identifier for connection
            connection_id = f"{from_edge}->{to_edge}"

            # Using details from <edge> if available, else setting defaults
            details_from = edge_detail_map.get(from_edge, {'length': 0, 'speed_limit': 0})
            details_to = edge_detail_map.get(to_edge, {'length': 0, 'speed_limit': 0})

            edges[connection_id] = {
                'from_edge': from_edge,
                'to_edge': to_edge,
                'from_length': details_from['length'],
                'to_length': details_to['length'],
                'length': details_from['length'] + details_to['length'],
                'speed_limit': min(details_from['speed_limit'], details_to['speed_limit']),
                'time': (details_from['length'] + details_to['length']) / min(details_from['speed_limit'],
                                                                              details_to['speed_limit'])
                if min(details_from['speed_limit'], details_to['speed_limit']) != 0 else float('inf')
            }
        end_time = time.time()  # End timing for the connections
        print(f"Finished constructing connections in {end_time - start_time:.2f} seconds.")
        return edges

    def map_speeds_to_lanes(self, filename, edges):
        df = pd.read_csv(filename)

        # Create a dictionary for fast lookup of mean_speed based on edge_id
        speed_lookup = df.set_index('edge_id')['mean_speed'].to_dict()

        updated_edges = set()

        for connection_id, edge_data in tqdm(edges.items(), desc="Processing edges"):
            from_edge = edge_data['from_edge']
            to_edge = edge_data['to_edge']

            if from_edge in speed_lookup:
                edges[connection_id]['from_speed'] = speed_lookup[from_edge]
                updated_edges.add(from_edge)

            if to_edge in speed_lookup:
                edges[connection_id]['to_speed'] = speed_lookup[to_edge]
                updated_edges.add(to_edge)

            # If both the from_edge and to_edge have their speed updated, compute the time
            if from_edge in updated_edges and to_edge in updated_edges:
                from_speed = edges[connection_id].get('from_speed', edge_data['speed_limit'])
                to_speed = edges[connection_id].get('to_speed', edge_data['speed_limit'])

                # The time is a weighted average based on the lengths and speeds of the individual edges
                from_length = edge_data['from_length']
                to_length = edge_data['to_length']

                time_from = from_length / from_speed if from_speed != 0 else float('inf')
                time_to = to_length / to_speed if to_speed != 0 else float('inf')

                edges[connection_id]['time'] = time_from + time_to
                print(edges[connection_id])

    def build_graph_from_edges(self, edges):
        # Initialize a directed graph
        G = nx.DiGraph()

        # Prepare edge data for batch insertion
        edge_data = []
        for connection_id, data in tqdm(edges.items(), desc="Adding edges to graph"):
            # Create a single tuple for each edge (start, end, attr_dict)
            attr_dict = {
                'weight': data.get('time', float('inf')),
                'connection_id': connection_id,
                'length': data['length']
            }

            if 'from_speed' in data and 'to_speed' in data:
                attr_dict['from_speed'] = data['from_speed']
                attr_dict['to_speed'] = data['to_speed']

            edge_data.append((data['from_edge'], data['to_edge'], attr_dict))

        # Add all edges to the graph in a batch
        G.add_edges_from(edge_data)

        return G

    def constrained_shortest_path(self, source_edge, target_edge, max_time, new_df):
        self.update_edge_speed(new_df)
        # Find the shortest path with a constraint on maximum time
        path = self.dijkstra_with_max_time(self.G, source_edge, target_edge, max_time)
        # path.insert(0, self.edges[source_edge]['from'])
        # path.append(self.edges[target_edge]['to'])
        print(f"Path from dijkstra_with_max_time: {path}")  # Debug line
        if not path:
            path = nx.shortest_path(self.G, source_edge, target_edge, weight='weight')
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

        print("Starting Dijkstra's with max time...")
        print(f"Source: {source}, Target: {target}, Max Time: {max_time}")

        # Main loop for Dijkstra's algorithm
        while to_explore:
            accumulated_time, current = min(to_explore)
            to_explore.remove((accumulated_time, current))

            # print(f"Exploring node {current} with distance {d}")

            if current == target and accumulated_time <= max_time:
                print(f"Found target {target} with distance {accumulated_time}")
                # Construct the path from source to target
                path = []
                while current:
                   # print(f"Adding {current} to the path")  # Add this debug line
                    path.append(current)
                    current = pred[current]
                    if current == source:  # Add this check
                        print("Reached the source node in path reconstruction")
                print("Constructed path:", path)  # Add this debug line
                return path[::-1]

            for neighbor in G[current]:
                new_dist = accumulated_time + G[current][neighbor]['weight']
                if new_dist < dist[neighbor] and new_dist <= max_time:
                    dist[neighbor] = new_dist
                    pred[neighbor] = current
                    to_explore.append((new_dist, neighbor))
                    # print(f"Updated distance for node {neighbor}: {new_dist}")
                elif new_dist > max_time:
                    print(f"Node {neighbor}'s distance {new_dist} exceeds max_time")

        print("Couldn't find a valid path within max_time.")
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

    def analyze_route(self, source_edge, target_edge, max_time_limit):
        # Get the 'to' node of the source edge and the 'from' node of the target edge
        source_node = self.edges[source_edge]['to']
        target_node = self.edges[target_edge]['from']

        # Analyze the best route from source to target with a maximum time constraint
        path = self.constrained_shortest_path(source_node, target_node, max_time_limit)
        if path:
            path.insert(0, self.edges[source_edge]['from'])
            path.append(target_node)
        self.plot_graph_with_path(path)
        return self.dijkstra_top_three_routes(source_node, target_node, max_time_limit)

    # def csv2df(self, csv_path):
    #     # Read the new CSV file
    #     df = pd.read_csv(csv_path)
    #     return df

    def update_edge_speed(self, df):

        print(df)

        # Create a dictionary for fast lookup of mean_speed based on edge_id
        speed_lookup = df.set_index('edge_id')['mean_speed'].to_dict()

        updated_edges = set()

        for connection_id, edge_data in tqdm(self.edges.items(), desc="Updating edge speeds"):
            from_edge = edge_data['from_edge']
            to_edge = edge_data['to_edge']

            if from_edge in speed_lookup:
                self.edges[connection_id]['from_speed'] = speed_lookup[from_edge]
                updated_edges.add(from_edge)

            if to_edge in speed_lookup:
                self.edges[connection_id]['to_speed'] = speed_lookup[to_edge]
                updated_edges.add(to_edge)

            # If both the from_edge and to_edge have their speed updated, compute the time
            if from_edge in updated_edges and to_edge in updated_edges:
                from_speed = self.edges[connection_id].get('from_speed', edge_data['speed_limit'])
                to_speed = self.edges[connection_id].get('to_speed', edge_data['speed_limit'])

                # The time is a weighted average based on the lengths and speeds of the individual edges
                from_length = edge_data['from_length']
                to_length = edge_data['to_length']

                time_from = from_length / from_speed if from_speed != 0 else float('inf')
                time_to = to_length / to_speed if to_speed != 0 else float('inf')

                self.edges[connection_id]['time'] = time_from + time_to

            # Update the graph (NetworkX) to reflect this change
            if (edge_data['from_edge'], edge_data['to_edge']) in self.G.edges:
                self.G[edge_data['from_edge']][edge_data['to_edge']]['weight'] = self.edges[connection_id]['time']






