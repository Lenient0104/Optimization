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
        for edge in edges:
            G.add_node(edge, length=self.edge_map[edge]['length'])
        # Add edges to the graph with weights
        for connection in self.connections:
            G.add_edge(connection[0], connection[1], travel_times={
                'e_bike_1': 10,
                'e_car': 5,
                'e_scooter_1': 8,
                'walking': 20
            })
        return G

    def shortest_path_for_mode(self, source, target, mode):
        G = self.build_graph()

        def mode_weight(u, v, d):
            return d['travel_times'].get(mode, float('inf'))

        path = nx.shortest_path(G, source, target, weight=mode_weight)
        return path

    def classify_stations(self):
        station_edges = self.get_stations()  # Retrieve the current station assignments for each edge
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
        print(mode_stations)
        return mode_stations

    # def build_e_bike_subgraph(self, G):
    #     # Initialize a new directed graph for e-bike stations
    #     e_bike_G = nx.DiGraph()
    #
    #     # Iterate over all edges in your original graph
    #     for u, v, data in G.edges(data=True):
    #         if 'e_bike_1' in data['travel_times']:
    #             # Add both the nodes and the edge to the e-bike graph if not already added
    #             if not e_bike_G.has_node(u):
    #                 e_bike_G.add_node(u)
    #             if not e_bike_G.has_node(v):
    #                 e_bike_G.add_node(v)
    #             # Add the edge with the e-bike travel time
    #             e_bike_G.add_edge(u, v, weight=data['travel_times']['e_bike_1'])
    #
    #     return e_bike_G

    # def calculate_all_e_bike_shortest_paths(self, G):
    #     # Calculate shortest paths for all pairs of nodes in the e-bike subgraph
    #     all_pairs_shortest_path = dict(nx.all_pairs_dijkstra_path(G, weight='weight'))
    #     return all_pairs_shortest_path
    #
    # def calculate_e_bike_paths(self):
    #     # Build the original graph with all modes and connections
    #     G = self.build_graph()
    #
    #     # Build the e-bike subgraph
    #     e_bike_G = self.build_e_bike_subgraph(G)
    #
    #     # Calculate and return all e-bike shortest paths
    #     return self.calculate_all_e_bike_shortest_paths(e_bike_G)

    def calculate_mode_shortest_paths(self, mode):
        # Ensure the graph is built with all the necessary data
        G = self.build_graph()

        # Assuming you have a method or a data structure that gives you stations for the mode
        mode_stations = self.classify_stations()[mode]

        # Initialize a structure to hold the shortest paths
        shortest_paths = {}

        # Calculate shortest paths between all pairs of mode-specific stations
        for source in mode_stations:
            for target in mode_stations:
                if source != target:  # Avoid calculating path to itself
                    try:
                        path = self.shortest_path_for_mode(source, target, mode)
                        shortest_paths[(source, target)] = path
                    except nx.NetworkXNoPath:
                        print(f"No path found from {source} to {target} for mode {mode}.")

        return shortest_paths





