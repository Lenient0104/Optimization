import sqlite3
import time
from aco import Ant
from e_car import ECar_EnergyConsumptionModel
from e_bike import Ebike_PowerConsumptionCalculator
from e_scooter import Escooter_PowerConsumptionCalculator
from user_info import User
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from tqdm import tqdm


class Optimization:
    def __init__(self, net_xml_path, user, db_path):
        self.user = user
        self.net_xml_path = net_xml_path
        self.db_connection = sqlite3.connect(db_path)
        # self.edges_csv_path = edges_csv_path
        self.edge_mapping = self.extract_edge_info()
        # Parse the XML file to get road and lane information
        self.edges = self.parse_net_xml(net_xml_path)
        # Map time to lanes using the provided CSV file
        self.mode_time_calculation()
        # Insert energy model
        self.ecar_energymodel = ECar_EnergyConsumptionModel(4)
        self.ebike_energymodel = Ebike_PowerConsumptionCalculator()
        self.escooter_energymodel = Escooter_PowerConsumptionCalculator()
        self.map_energy_to_lanes(60, 1)
        self.map_station_availability()
        self.initialize_pheromone_levels()

        # Build a graph based on the edge data
        # self.G = self.build_graph_from_edges(self.edges)

    # def user_thread(self, weight, driving_license, pal):
    #     user = User(weight, driving_license, pal)
    #     return user

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
                    'e_bike_1': {'from_speed': 0, 'to_speed': 0},
                    'e_car': {'from_speed': 0, 'to_speed': 0},
                    'e_scooter_1': {'from_speed': 0, 'to_speed': 0},
                    'walking': {'speed': 1.4}
                },
                'from_average_speed': 0,
                'to_average_speed': 0,
                'energy_consumption': {
                    'e_bike_1': {'from_energy': 0, 'to_energy': 0},
                    'e_car': {'from_energy': 0, 'to_energy': 0},
                    'e_scooter_1': {'from_energy': 0, 'to_energy': 0}
                },
                'station_availability': {
                    'from': {'e_bike_1': False, 'e_car': False, 'e_scooter_1': False, 'walking': True},
                    'to': {'e_bike_1': False, 'e_car': False, 'e_scooter_1': False, 'walking': True}
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
        return edges

    def mode_time_calculation(self):
        # speed limit:
        # e_scooter_1 = 6.9, e_bike_1 = 6.9, e_car = 41.7 m/s
        cursor = self.db_connection.cursor()
        query = """
                SELECT EdgeID, Speed
                FROM EdgeSpeed
                WHERE TimeStep = (SELECT MAX(TimeStep) FROM EdgeSpeed)
                """
        cursor.execute(query)
        latest_speed_data = cursor.fetchall()
        # Create a dictionary for fast lookup of mean_speed based on edge_id
        speed_lookup = {edge_id: speed for edge_id, speed in latest_speed_data}

        edges = self.edges

        # Update the speed in the edges dictionary
        for connection_id, edge_data in self.edges.items():
            from_edge = int(edge_data['from_edge'])
            to_edge = int(edge_data['to_edge'])
            from_length = edge_data['from_length']
            to_length = edge_data['to_length']

            if from_edge in speed_lookup:
                edges[connection_id]['from_average_speed'] = speed_lookup[from_edge]
                from_average_speed = edges[connection_id]['from_average_speed']
                if from_average_speed < 6.9:
                    edges[connection_id]['mode_time']['e_bike_1']['from_time'] = from_length / from_average_speed
                    edges[connection_id]['mode_time']['e_scooter_1']['from_time'] = from_length / from_average_speed
                    edges[connection_id]['mode_time']['e_car']['from_time'] = from_length / from_average_speed
                    edges[connection_id]['mode_time']['walking']['from_time'] = from_length / 1.4

                    edges[connection_id]['speed']['e_bike_1']['from_speed'] = from_average_speed
                    edges[connection_id]['speed']['e_scooter_1']['from_speed'] = from_average_speed
                    edges[connection_id]['speed']['e_car']['from_speed'] = from_average_speed

                if 6.9 <= from_average_speed < 41.7:
                    edges[connection_id]['mode_time']['e_bike_1']['from_time'] = from_length / 6.9
                    edges[connection_id]['mode_time']['e_scooter_1']['from_time'] = from_length / 6.9
                    edges[connection_id]['mode_time']['e_car']['from_time'] = from_length / from_average_speed
                    edges[connection_id]['mode_time']['walking']['from_time'] = from_length / 1.4

                    edges[connection_id]['speed']['e_bike_1']['from_speed'] = 6.9
                    edges[connection_id]['speed']['e_scooter_1']['from_speed'] = 6.9
                    edges[connection_id]['speed']['e_car']['from_speed'] = from_average_speed
                else:
                    edges[connection_id]['mode_time']['e_bike_1']['from_time'] = from_length / 6.9
                    edges[connection_id]['mode_time']['e_scooter_1']['from_time'] = from_length / 6.9
                    edges[connection_id]['mode_time']['e_car']['from_time'] = from_length / 41.7
                    edges[connection_id]['mode_time']['walking']['from_time'] = from_length / 1.4

                    edges[connection_id]['speed']['e_bike_1']['from_speed'] = 6.9
                    edges[connection_id]['speed']['e_scooter_1']['from_speed'] = 6.9
                    edges[connection_id]['speed']['e_car']['from_speed'] = 41.7

            if to_edge in speed_lookup:
                edges[connection_id]['to_average_speed'] = speed_lookup[to_edge]
                to_average_speed = edges[connection_id]['to_average_speed']
                if to_average_speed < 6.9:
                    edges[connection_id]['mode_time']['e_bike_1']['to_time'] = to_length / to_average_speed
                    edges[connection_id]['mode_time']['e_scooter_1']['to_time'] = to_length / to_average_speed
                    edges[connection_id]['mode_time']['e_car']['to_time'] = to_length / to_average_speed
                    edges[connection_id]['mode_time']['walking']['to_time'] = from_length / 1.4

                    edges[connection_id]['speed']['e_bike_1']['to_speed'] = to_average_speed
                    edges[connection_id]['speed']['e_scooter_1']['to_speed'] = to_average_speed
                    edges[connection_id]['speed']['e_car']['to_speed'] = to_average_speed
                if 6.9 <= to_average_speed < 41.7:
                    edges[connection_id]['mode_time']['e_bike_1']['to_time'] = to_length / 6.9
                    edges[connection_id]['mode_time']['e_scooter_1']['to_time'] = to_length / 6.9
                    edges[connection_id]['mode_time']['e_car']['to_time'] = to_length / to_average_speed
                    edges[connection_id]['mode_time']['walking']['to_time'] = from_length / 1.4

                    edges[connection_id]['speed']['e_bike_1']['to_speed'] = 6.9
                    edges[connection_id]['speed']['e_scooter_1']['to_speed'] = 6.9
                    edges[connection_id]['speed']['e_car']['to_speed'] = to_average_speed

                else:
                    edges[connection_id]['mode_time']['e_bike_1']['to_time'] = to_length / 6.9
                    edges[connection_id]['mode_time']['e_scooter_1']['to_time'] = to_length / 6.9
                    edges[connection_id]['mode_time']['e_car']['to_time'] = to_length / 41.7
                    edges[connection_id]['mode_time']['walking']['to_time'] = from_length / 1.4

                    edges[connection_id]['speed']['e_bike_1']['to_speed'] = 6.9
                    edges[connection_id]['speed']['e_scooter_1']['to_speed'] = 6.9
                    edges[connection_id]['speed']['e_car']['to_speed'] = 41.7

    def map_energy_to_lanes(self, user_weight, user_pal):
        edges = self.edges
        for connection_id, edge_data in tqdm(edges.items(), desc="Processing edges"):
            # Speed limits for different modes
            ebike_speed_limit = 6.9  # in m/s
            ecar_speed_limit = 41.7  # in m/s
            escooter_speed_limit = 6.9  # in m/s
            walking_speed = 1.4  # in m/s

            # Retrieve the speeds for each mode from the edge data
            from_ebike_speed = min(edge_data['speed']['e_bike_1']['from_speed'], ebike_speed_limit)
            to_ebike_speed = min(edge_data['speed']['e_bike_1']['to_speed'], ebike_speed_limit)
            from_ecar_speed = min(edge_data['speed']['e_car']['from_speed'], ecar_speed_limit)
            to_ecar_speed = min(edge_data['speed']['e_car']['to_speed'], ecar_speed_limit)
            from_escooter_speed = min(edge_data['speed']['e_scooter_1']['from_speed'], escooter_speed_limit)
            to_escooter_speed = min(edge_data['speed']['e_scooter_1']['to_speed'], escooter_speed_limit)

            # Calculate energy consumption for e_bike_1, e_car, e_scooter_1
            edge_data['energy_consumption']['e_bike_1']['from_energy'] = self.ebike_energymodel.calculate(
                from_ebike_speed, user_weight, 1, user_pal)*edge_data['from_length']
            edge_data['energy_consumption']['e_bike_1']['to_energy'] = self.ebike_energymodel.calculate(to_ebike_speed,
                                                                                                      user_weight, 1,
                                                                                                      user_pal)*edge_data['to_length']

            edge_data['energy_consumption']['e_car']['from_energy'] = self.ecar_energymodel.calculate_energy_loss(
                from_ecar_speed)*edge_data['from_length']
            edge_data['energy_consumption']['e_car']['to_energy'] = self.ecar_energymodel.calculate_energy_loss(
                to_ecar_speed)*edge_data['to_length']


            edge_data['energy_consumption']['e_scooter_1']['from_energy'] = self.escooter_energymodel.calculate(
                from_escooter_speed, user_weight, 1)*edge_data['from_length']
            edge_data['energy_consumption']['e_scooter_1']['to_energy'] = self.escooter_energymodel.calculate(
                to_escooter_speed, user_weight, 1)*edge_data['to_length']

    def map_station_availability(self):
        # static information
        cursor = self.db_connection.cursor()
        query = "SELECT StationEdgeID, StationType FROM StationLocation"
        cursor.execute(query)
        station_data = cursor.fetchall()
        print(station_data)

        for station_edge_id, station_type in station_data:
            for connection_id, edge_data in self.edges.items():
                if edge_data['from_edge'] == str(station_edge_id):
                    self.edges[connection_id]['station_availability']['from'][station_type] = True
                if edge_data['to_edge'] == str(station_edge_id):
                    self.edges[connection_id]['station_availability']['to'][station_type] = True

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


    def convert(self, edges):

        new_structure = {}
        for connection_id, edge_info in edges.items():
            # Extract 'from' and 'to' station names
            from_edge = edge_info['from_edge']
            to_edge = edge_info['to_edge']

            new_structure[from_edge] = {}

            # Iterate through each mode of transport
            for mode in edge_info['mode_time']:
                # Calculate the total time cost for each mode
                from_time = edge_info['mode_time'][mode]['from_time']
                to_time = edge_info['mode_time'][mode]['to_time']
                total_time = from_time + to_time

                # Calculate the total energy cost for each mode
                if mode in edge_info['energy_consumption']:
                    from_energy = edge_info['energy_consumption'][mode]['from_energy']
                    to_energy = edge_info['energy_consumption'][mode]['to_energy']
                    total_energy = from_energy + to_energy
                else:
                    total_energy = 0  # Default to 0 if the mode is not in energy_consumption

                # Initialize the mode of transport for the station if not already present
                if mode not in new_structure[from_edge]:
                    new_structure[from_edge][mode] = []

                # Append the new tuple
                new_structure[from_edge][mode].append((to_edge, mode, total_energy, total_time))

        return new_structure

    # Function to find the best path
    def find_best_path(self, ants):
        best_ant = None
        best_time_cost = float('inf')
        for ant in ants:
            if ant.total_time_cost < best_time_cost:
                best_time_cost = ant.total_time_cost
                best_ant = ant
        return best_ant

    def run_aco_algorithm(self, start_edge, destination_edge):
        number_of_ants = 1  # Number of ants to use
        # Run the ACO algorithm
        number_of_iterations = 10
        log_interval = 1

        # Initialize ants with specified start and destination locations
        ants = [Ant(start_edge, destination_edge, 'e_scooter_1', self.db_connection, self.edges) for _ in
                range(number_of_ants)]
        best_path = None
        best_time_cost = float('inf')

        for iteration in range(number_of_iterations):
            print(f"Starting iteration {iteration + 1}")
            for ant in ants:
                move_count = 0
                while ant.path[-1][0] != destination_edge:
                    ant.move()
                    print(
                        "-------one move--------")
                    move_count += 1
                    if move_count > 100:
                        print("Move limit reached, breaking out of loop.")
                        break

                print("------one ant-------time cost:", ant.total_time_cost)

            current_best_ant = self.find_best_path(ants)  # based on time cost
            if current_best_ant and current_best_ant.total_time_cost < best_time_cost:
                best_path = current_best_ant.path
                best_time_cost = current_best_ant.total_time_cost

            ants = [Ant(start_edge, destination_edge, 'e_scooter_1', self.db_connection, self.edges) for _ in
                    range(number_of_ants)]

            if (iteration + 1) % log_interval == 0 or iteration == number_of_iterations - 1:
                print(
                    f"Iteration {iteration + 1}/{number_of_iterations}: Best Path = {best_path}, Time Cost = {best_time_cost}")

        print("Best Path after all iterations:", best_path)
        return best_path
