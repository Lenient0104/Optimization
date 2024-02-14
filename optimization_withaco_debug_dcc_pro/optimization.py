import heapq
import re
import sqlite3
import time
import random
import networkx as nx

from aco import Ant
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyvis.network import Network
from heapq import heappush, heappop
from collections import defaultdict
from e_car import ECar_EnergyConsumptionModel
from e_bike import Ebike_PowerConsumptionCalculator
from e_scooter import Escooter_PowerConsumptionCalculator


class Optimization:
    def __init__(self, net_xml_path, user, db_path):
        self.user = user
        self.unique_edges = []
        self.connections = []
        self.net_xml_path = net_xml_path
        self.db_connection = sqlite3.connect(db_path)

        # Parse the XML file to get road and lane information
        self.edges, self.edge_map = self.parse_net_xml(net_xml_path)
        # Map time to lanes using the provided CSV file
        self.edges_station = self.get_stations()
        self.map_speed()
        self.mode_time_cal()
        # Insert energy model
        self.ecar_energymodel = ECar_EnergyConsumptionModel(4)
        self.ebike_energymodel = Ebike_PowerConsumptionCalculator()
        self.escooter_energymodel = Escooter_PowerConsumptionCalculator()
        # self.map_energy_to_lanes(60, 1)
        # self.map_station_availability()

        self.initialize_pheromone_levels()
        self.G = self.build_graph()

    def parse_net_xml(self, filepath):
        pattern = r"^[A-Za-z]+"
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
                    'shape': lane.attrib['shape'],
                }
        end_time = time.time()  # End timing
        print(f"Finished information extraction from edges in {end_time - start_time:.2f} seconds.")

        start_time = time.time()
        # Constructing edges from <connection> tags
        for conn in tqdm(root.findall('connection'), desc="Processing connections", mininterval=1.0):
            pairs = []
            from_edge = conn.get('from')
            if from_edge.startswith(":") or re.match(pattern, from_edge) is not None:
                break
            if from_edge not in self.unique_edges and from_edge != 'gneE29':
                self.unique_edges.append(from_edge)
            to_edge = conn.get('to')
            if to_edge not in self.unique_edges:
                self.unique_edges.append(to_edge)

            connection_id = f"{from_edge}->{to_edge}"
            pairs.append(from_edge)
            pairs.append(to_edge)
            self.connections.append(pairs)

            # Using details from <edge> if available, else setting defaults
            details_from = edge_detail_map.get(from_edge, {'length': 0, 'speed_limit': 0})
            details_to = edge_detail_map.get(to_edge, {'length': 0, 'speed_limit': 0})

            # edges data structure
            edges[connection_id] = {
                'from_edge': from_edge,
                'to_edge': to_edge,
                'from_length': details_from['length'],
                'to_length': details_to['length'],
                'length': details_from['length'] + details_to['length'],
                'speed_limit': 13.9,
                'mode_time': {
                    'e_bike_1': {'from_time': 0, 'to_time': 0},
                    'e_car': {'from_time': 0, 'to_time': 0},
                    'e_scooter_1': {'from_time': 0, 'to_time': 0},
                    'walking': {'from_time': 0, 'to_time': 0}
                },
                'speed': {
                    'e_bike_1': {'from_speed': 0.0, 'to_speed': 0.0},
                    'e_car': {'from_speed': 0.0, 'to_speed': 0.0},
                    'e_scooter_1': {'from_speed': 0.0, 'to_speed': 0.0},
                    'walking': {'speed': 1.4}
                },
                'from_average_speed': 0,
                'to_average_speed': 0,
                'energy_consumption': {
                    'e_bike_1': {'from_energy': 0, 'to_energy': 0},
                    'e_car': {'from_energy': 0, 'to_energy': 0},
                    'e_scooter_1': {'from_energy': 0, 'to_energy': 0}
                },
                'pheromone_levels': {
                    'e_bike_1': {'from': 0.1, 'to': 0.1},
                    'e_car': {'from': 0.1, 'to': 0.1},
                    'e_scooter_1': {'from': 0.1, 'to': 0.1},
                    'walking': {'from': 0.1, 'to': 0.1}
                }
            }

        end_time = time.time()  # End timing for the connections
        print(f"Finished constructing connections in {end_time - start_time:.2f} seconds.")
        return edges, edge_detail_map

    def map_speed(self):
        for _, edge_data in tqdm(self.edges.items(), desc="Setting Speed Data"):
            edge_data['speed']['e_bike_1']['from_speed'] = 5.6
            edge_data['speed']['e_bike_1']['to_speed'] = 5.6
            edge_data['speed']['e_scooter_1']['from_speed'] = 5.6
            edge_data['speed']['e_scooter_1']['to_speed'] = 5.6
            edge_data['speed']['e_car']['from_speed'] = 33.3
            edge_data['speed']['e_car']['to_speed'] = 33.3

    def mode_time_cal(self):
        for _, edge_data in tqdm(self.edges.items(), desc="Setting Time Data"):
            edge_data['mode_time']['e_bike_1']['from_time'] = edge_data['from_length'] / edge_data['speed']['e_bike_1'][
                'from_speed']
            edge_data['mode_time']['e_bike_1']['to_time'] = edge_data['to_length'] / edge_data['speed']['e_bike_1'][
                'to_speed']
            edge_data['mode_time']['e_scooter_1']['from_time'] = edge_data['from_length'] / \
                                                                 edge_data['speed']['e_scooter_1'][
                                                                     'from_speed']
            edge_data['mode_time']['e_scooter_1']['to_time'] = edge_data['to_length'] / \
                                                               edge_data['speed']['e_scooter_1'][
                                                                   'to_speed']
            edge_data['mode_time']['e_car']['from_time'] = edge_data['from_length'] / edge_data['speed']['e_car'][
                'from_speed']
            edge_data['mode_time']['e_car']['to_time'] = edge_data['to_length'] / edge_data['speed']['e_car'][
                'to_speed']
            edge_data['mode_time']['walking']['from_time'] = edge_data['from_length'] / edge_data['speed']['walking'][
                'speed']
            edge_data['mode_time']['walking']['to_time'] = edge_data['to_length'] / edge_data['speed']['walking'][
                'speed']

    # Define a function to generate random stations for an edge
    # def generate_random_stations(self):
    #     # Initialize a list of stations with 'walking' station
    #     stations = ['walking']
    #     station_types = ['e_bike_1', 'e_scooter_1', 'e_car']
    #
    #     # Randomly decide whether to add other station types
    #     for station_type in station_types:
    #         if random.choice([True, False]):
    #             stations.append(station_type)
    #     print(stations)
    #     return stations
    #
    # def get_stations(self):
    #     # Create a dictionary to store stations for each edge
    #     edge_stations = {}
    #
    #     # Randomly select up to 10 edges if there are more than 10, else select all
    #     selected_edges = random.sample(self.unique_edges, min(len(self.unique_edges), 10))
    #
    #     # Initialize all edges with just 'walking' station
    #     for edge_id in self.unique_edges:
    #         edge_stations[edge_id] = ['walking']
    #
    #     # Generate stations for the selected edges
    #     for edge_id in selected_edges:
    #         edge_stations[edge_id] = self.generate_random_stations()
    #
    #     return edge_stations

    def get_stations(self):
        # Ensure all edges initially have just a 'walking' station
        edge_stations = {edge_id: ['walking'] for edge_id in self.unique_edges}

        # Determine the number of edges to assign e-mobility stations to (up to 10)
        num_edges_to_assign = min(len(self.unique_edges), 10)

        # Select random edges to assign e-mobility stations
        edges_to_assign = random.sample(self.unique_edges, num_edges_to_assign)

        # Assign at least one e-mobility station type to each selected edge
        for edge_id in edges_to_assign:
            # Choose at least one e-mobility station type for this edge
            e_mobility_stations = random.sample(['e_bike_1', 'e_scooter_1', 'e_car'], k=1)
            # There's an opportunity to add more e-mobility stations randomly
            for station in ['e_bike_1', 'e_scooter_1', 'e_car']:
                if station not in e_mobility_stations and random.choice([True, False]):
                    e_mobility_stations.append(station)
            # Assign the selected e-mobility stations to this edge
            edge_stations[edge_id] = ['walking'] + e_mobility_stations

        return edge_stations

    def initialize_pheromone_levels(self):
        for connection_id in self.edges:
            # Initialize pheromone levels for each transportation mode
            pheromone_init_value = 0.1
            self.edges[connection_id]['pheromone_levels'] = {
                'e_bike_1': {'from': pheromone_init_value, 'to': pheromone_init_value},
                'e_car': {'from': pheromone_init_value, 'to': pheromone_init_value},
                'e_scooter_1': {'from': pheromone_init_value, 'to': pheromone_init_value},
                'walking': {'from': pheromone_init_value, 'to': pheromone_init_value}
            }
        # print(self.convert(self.edges)['-1'])

    def check_edge_existence(self, edge):
        if edge in self.unique_edges:
            return True
        return False

    def visualization(self, time_cost, ant_num):
        # Generating the iteration numbers
        iterations = list(range(1, len(time_cost) + 1))
        y_value = []
        for time in time_cost:
            y_value.append(time - min(time_cost))

        # y_value.append(0)
        # y_value.append(0)

        # Creating the plot
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, y_value, marker='o', alpha=0.5)
        plt.title(f"Optimality of Objective Value Over Iterations ({ant_num} Ants Used)")
        plt.xlabel("Number of Iterations")
        plt.ylabel("Time Cost (seconds) Difference from the Best Time Cost")
        plt.grid(True)
        plt.show()

    # Function to find the best path
    def find_best_path(self, ants, destination_edge, pheromones):
        best_ant = None
        best_phe = 0
        for ant in ants:
            # Check if the ant has reached the destination
            if ant.path[-1][0] == destination_edge:
                # If the ant has reached the destination and has a lower total time cost
                if pheromones[ant] > best_phe:
                    best_phe = pheromones[ant]
                    best_ant = ant

        return best_ant

    def update_pheromones(self, ant):
        total_pheromone = 0
        for i in range(0, len(ant.path) - 1):
            edge_1, mode_1, _, time_1 = ant.path[i]
            edge_2, mode_2, _, time_2 = ant.path[i + 1]
            key = str(edge_1) + '->' + str(edge_2)
            if i != 0:
                self.edges[(key)]['pheromone_levels'][mode_1]['from'] += self.pheromone_deposit_function(
                    ant.total_time_cost)
            self.edges[(key)]['pheromone_levels'][mode_2]['to'] += self.pheromone_deposit_function(
                ant.total_time_cost)
            total_pheromone = total_pheromone + self.edges[(key)]['pheromone_levels'][mode_2]['to']
        return total_pheromone

    def pheromone_deposit_function(self, path_cost):
        # Higher deposit for faster paths
        return 1 / path_cost  # Example function, adjust as needed

    def build_graph(self):
        G = nx.DiGraph()
        edges = self.unique_edges
        flag = '3789374#3' in edges
        flag_4 = '41592114' in edges
        for edge in edges:
            G.add_node(edge, length=self.edge_map[edge]['length'])
        flag_1 = G.has_node('3789374#3')
        flag_2 = G.has_node('41592114')
        # Add edges to the graph with weights
        for connection in self.connections:
            G.add_edge(connection[0], connection[1], travel_times={
                'e_bike_1': 10,
                'e_car': 5,
                'e_scooter_1': 8,
                'walking': 20
            })
        print(G.number_of_nodes(), G.number_of_edges())
        return G

    def shortest_path_for_mode(self, source, target, mode):
        G = self.G

        def mode_weight(u, v, d):
            return d['travel_times'].get(mode, float('inf'))

        path = nx.shortest_path(G, source, target, weight=mode_weight)
        return path

    def classify_stations(self):
        station_edges = self.edges_station  # Retrieve the current station assignments for each edge
        mode_stations = {
            'e_bike_1': [],
            'e_scooter_1': [],
            'e_car': []
        }

        # Iterate through each edge and its assigned stations
        for edge_id, stations in station_edges.items():
            # Check for the presence of each station type and classify the edge accordingly
            if 'e_bike_1' in stations:
                mode_stations['e_bike_1'].append(edge_id)
            if 'e_scooter_1' in stations:
                mode_stations['e_scooter_1'].append(edge_id)
            if 'e_car' in stations:
                mode_stations['e_car'].append(edge_id)

        # Optionally, print the classification result for debugging
        # print(mode_stations)
        return mode_stations

    def calculate_all_modes_shortest_paths(self):
        G = self.build_graph()  # Ensure this graph includes travel times correctly set up for each edge

        all_modes_shortest_paths = {}
        station_classifications = self.classify_stations()

        for mode, mode_stations in station_classifications.items():
            shortest_paths = {}

            # Define a custom weight function that extracts the correct travel time for the mode
            def weight(u, v, d):
                return d['travel_times'].get(mode, float('inf'))  # Use a large default value if mode is not found

            for source in mode_stations:
                for target in mode_stations:
                    if source != target:
                        try:
                            # Use the custom weight function for the shortest path calculation
                            path = nx.shortest_path(G, source=source, target=target, weight=weight)
                            # Calculate the total travel time for the path
                            total_time = sum(
                                G[path[i]][path[i + 1]]['travel_times'][mode] for i in range(len(path) - 1))
                            shortest_paths[(source, target)] = (path, total_time)
                        except nx.NetworkXNoPath:
                            print(f"No path found from {source} to {target} for mode {mode}.")

            all_modes_shortest_paths[mode] = shortest_paths

        return all_modes_shortest_paths

    def calculate_walking_paths_from_start(self, source, destination_edge):
        G = self.G  # Ensure this graph includes walking times as weights

        def walking_time(u, v, d):
            return d['travel_times']['walking']

        walking_paths = {}
        station_classifications = self.classify_stations()
        all_station_edges = set().union(*station_classifications.values(), {destination_edge})

        for target in all_station_edges:
            if source != target:  # Avoid calculating path to itself
                try:
                    path = nx.shortest_path(G, source=source, target=target, weight=walking_time)
                    total_time = sum(G[path[i]][path[i + 1]]['travel_times']['walking'] for i in range(len(path) - 1))
                    walking_paths[target] = (path, total_time)
                except nx.NodeNotFound:
                    print(f"Node not found from {source} to {target}")
                except nx.NetworkXNoPath:
                    print(f"No walking path found from {source} to {target}.")

        return walking_paths

    def calculate_walking_paths_to_destination(self, start_edge, destination_edge):
        G = self.G  # Ensure this graph includes walking times as weights

        # Extract the walking time for an edge
        def walking_time(u, v, d):
            return d['travel_times']['walking']

        # Initialize a structure to hold the shortest paths and their total time costs
        paths_to_destination = {}

        # Get station classifications
        station_classifications = self.classify_stations()

        # Aggregate all unique station edges across all modes, including the start edge for completeness
        all_station_edges = set().union(*station_classifications.values(), {start_edge})

        # Calculate paths from all station edges and the start edge to the destination edge using walking mode
        for source in all_station_edges:
            if source != destination_edge:  # Exclude path from the destination to itself
                try:
                    path = nx.shortest_path(G, source=source, target=destination_edge, weight=walking_time)
                    # Calculate the total walking time for the path
                    total_time = sum(G[path[i]][path[i + 1]]['travel_times']['walking'] for i in range(len(path) - 1))
                    paths_to_destination[source] = (path, total_time)
                except nx.NodeNotFound:
                    print(f"Node not found from {source} to {destination_edge}.")
                except nx.NetworkXNoPath:
                    print(f"No walking path found from {source} to {destination_edge}.")

        return paths_to_destination

    def build_all_modes_paths_graph_corrected(self, start_edge, destination_edge):
        paths_graph = nx.MultiDiGraph()

        # Collect all station edges
        station_classifications = self.classify_stations()
        all_station_edges = set().union(*station_classifications.values(), {start_edge, destination_edge})
        for node in all_station_edges:
            paths_graph.add_node(node)

        # Use the adjusted calculate_all_modes_shortest_paths that provides paths and total time costs
        all_modes_shortest_paths = self.calculate_all_modes_shortest_paths()

        # Add edges for each mode's paths with total time cost as weight
        for mode, mode_shortest_paths in all_modes_shortest_paths.items():
            for (source, target), (path, total_time) in mode_shortest_paths.items():
                if {source, target}.issubset(all_station_edges):
                    # Add a direct edge with total time cost as the weight
                    paths_graph.add_edge(source, target, weight=total_time, mode=mode)

        # Assume direct connections for walking paths with calculated total time costs
        walking_paths_from_start = self.calculate_walking_paths_from_start(start_edge, destination_edge)
        walking_paths_to_destination = self.calculate_walking_paths_to_destination(start_edge, destination_edge)

        # Add walking paths from start to all station edges and the destination with total time as weight
        for target, (path, total_time) in walking_paths_from_start.items():
            if target in all_station_edges:
                paths_graph.add_edge(start_edge, target, weight=total_time, mode='walking')

        # Add walking paths from all station edges and the start to the destination with total time as weight
        for source, (path, total_time) in walking_paths_to_destination.items():
            if source in all_station_edges:
                paths_graph.add_edge(source, destination_edge, weight=total_time, mode='walking')

        return paths_graph

    def visualize_paths_graph_interactive(self, paths_graph):
        # Initialize the Network object with cdn_resources set to 'remote' for better compatibility
        nt = Network(notebook=True, height="750px", width="100%", bgcolor="#222222", font_color="white",
                     cdn_resources='remote')

        # Define colors for different types of edges and nodes
        station_color = 'lightblue'
        start_end_color = 'green'
        walking_edge_color = 'lightgray'
        other_edge_color = 'blue'

        # Add nodes and edges with attributes
        for node in paths_graph.nodes:
            if node.startswith("station_"):
                nt.add_node(node, title=node, color=station_color)
            elif node in paths_graph.graph['start_nodes'] or node in paths_graph.graph['end_nodes']:
                nt.add_node(node, title=node, color=start_end_color)
            else:
                nt.add_node(node, title=node, color=other_edge_color)

        for edge in paths_graph.edges(data=True):
            src, dst, attr = edge
            title = f"Mode: {attr['mode']}"
            if attr['mode'] == 'walking':
                nt.add_edge(src, dst, title=title, color=walking_edge_color)
            else:
                nt.add_edge(src, dst, title=title, color=other_edge_color)

        # Configure the physics layout of the network
        nt.set_options(options="""{
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -80000,
              "centralGravity": 0.3,
              "springLength": 100,
              "springConstant": 0.04,
              "damping": 0.09,
              "avoidOverlap": 0.1
            },
            "minVelocity": 0.75
          }
        }""")

        # Show the interactive plot, specifying the path where the HTML file will be saved
        output_path = "paths_graph.html"
        nt.show(output_path)
        return output_path

    def visualize_paths_graph(self, paths_graph):
        plt.figure(figsize=(12, 12))  # Increase figure size
        pos = nx.kamada_kawai_layout(paths_graph)  # Use a different layout

        nx.draw_networkx_nodes(paths_graph, pos, node_size=500, node_color='lightblue')
        nx.draw_networkx_edges(paths_graph, pos, edge_color='gray', arrows=True)
        nx.draw_networkx_labels(paths_graph, pos, font_size=8)

        edge_labels = nx.get_edge_attributes(paths_graph, 'mode')
        nx.draw_networkx_edge_labels(paths_graph, pos, edge_labels=edge_labels, font_size=7)

        plt.title("Shortest Paths Graph Visualization")
        plt.axis('off')
        plt.show()


    def visualize_paths_graph_multi(self, paths_graph):
        pos = nx.spring_layout(paths_graph)  # Position nodes using a layout algorithm

        nx.draw(paths_graph, pos, with_labels=True, node_size=700, node_color='lightblue', edge_color='gray',
                arrows=True)

        # Simplify edge labels for MultiDiGraph: select one edge between each pair of nodes for labeling
        edge_labels = {}
        for u, v, data in paths_graph.edges(data=True):
            # Example: Select the first edge's 'mode' attribute as the label, or aggregate information
            if (u, v) not in edge_labels:  # Avoid overwriting if an edge label already exists for this node pair
                edge_labels[(u, v)] = f"{data.get('mode', '')}\n{data.get('weight', '')}"

        # Draw edge labels
        nx.draw_networkx_edge_labels(paths_graph, pos, edge_labels=edge_labels, font_size=7)

        plt.title("Paths Graph Visualization")
        plt.axis('off')  # Turn off the axis numbers / labels
        plt.show()

    def pre_computation(self, source, end):
        print("The path between station edges:")
        print(self.calculate_all_modes_shortest_paths())
        print("The walking paths from start to stations and end edge:")
        print(self.calculate_walking_paths_from_start(source, end))
        print("The walking paths from start and stations to end:")
        print(self.calculate_walking_paths_to_destination(source, end))
        new_graph = self.build_all_modes_paths_graph_corrected(source, end)
        # self.visualize_paths_graph_interactive(new_graph)
        self.visualize_paths_graph_multi(new_graph)
        print(new_graph.number_of_nodes(), new_graph.number_of_edges())
















