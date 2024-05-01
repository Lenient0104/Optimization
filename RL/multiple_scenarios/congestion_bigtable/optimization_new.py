import json
import random
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
import queryTest
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

from aco import Ant
from e_bike import Ebike_PowerConsumptionCalculator
from e_car import ECar_EnergyConsumptionModel
from e_scooter import Escooter_PowerConsumptionCalculator


class Optimization:
    def __init__(self, net_xml_path, user, db_path, simulation, start_edge, destination_edge):
        self.mode_stations = None
        self.new_graph = None
        self.user = user
        self.simulation = simulation
        self.unique_edges = []
        self.connections = []
        self.net_xml_path = net_xml_path
        self.db_connection = sqlite3.connect(db_path)

        # Parse the XML file to get road and lane information
        self.edge_map = self.parse_net_xml(net_xml_path)
        # Map time to lanes using the provided CSV file
        self.edges_station = self.get_stations(self.user)

        # Insert energy model
        self.ecar_energymodel = ECar_EnergyConsumptionModel(4)
        self.ebike_energymodel = Ebike_PowerConsumptionCalculator()
        self.escooter_energymodel = Escooter_PowerConsumptionCalculator()
        # self.map_energy_to_lanes(60, 1)
        # self.map_station_availability()


        self.G = self.build_graph()
        self.new_graph = self.build_new_graph(start_edge, destination_edge)

    def choose_od_pairs(self):
        random.seed(88)
        selected_pairs = random.sample(self.unique_edges, k=500 * 2)
        od_pairs = [(selected_pairs[i], selected_pairs[i + 1]) for i in range(0, len(selected_pairs), 2)]
        return od_pairs

    def parse_net_xml(self, filepath):
        pattern = r"^[A-Za-z]+"
        tree = ET.parse(filepath)
        root = tree.getroot()
        # Creating a map for edge details from <edge> tags
        edge_detail_map = {}

        for edge in tqdm(root.findall('edge'), desc="Processing edges"):
            for lane in edge.findall('lane'):
                edge_id = edge.attrib['id']
                edge_detail_map[edge_id] = {
                    'length': float(lane.attrib['length']),
                    'speed_limit': float(lane.attrib['speed']),
                    'shape': lane.attrib['shape'],
                }

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

        return edge_detail_map

    def get_stations(self, user):
        # Ensure all edges initially have just a 'walking' station
        edge_stations = {edge_id: ['walking'] for edge_id in self.unique_edges}

        edge_to_assign = ['361409608#3', '3791905#2', '-11685016#2', '369154722#2', '244844370#0', '37721356#0',
                          '74233405#1', '129774671#0', '23395388#5', '-64270141']
        # indices_to_assign = [50, 89, 112, 256, 309, 4000, 503, 8000, 30000, 10000]
        e_mobility_stations = user.preference
        if not user.driving_license and 'e_car' in user.preference:
            e_mobility_stations.remove('e_car')
        for edge_id in edge_to_assign:
            edge_stations[edge_id] = ['walking'] + e_mobility_stations
        return edge_stations

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
            edge_1, mode_1, _, _ = ant.path[i]
            edge_2, mode_2, _, _ = ant.path[i + 1]
            # Retrieve all edge data between edge_1 and edge_2
            all_edges_data = self.new_graph.get_edge_data(edge_1, edge_2)

            # Check if there are any edges between these nodes
            if all_edges_data:
                for key, edge_data in all_edges_data.items():
                    if key == mode_1:
                        # Correctly access and update pheromone level
                        current_pheromone_level = edge_data.get('pheromone_level',
                                                                0)  # Get current level, default to 0 if not set
                        # Update pheromone level
                        updated_pheromone_level = current_pheromone_level + self.pheromone_deposit_function(
                            ant.total_time_cost)
                        # Set the updated pheromone level back on the edge
                        self.new_graph[edge_1][edge_2][key]['pheromone_level'] = updated_pheromone_level

            total_pheromone = total_pheromone + self.new_graph[edge_1][edge_2][key]['pheromone_level']
        return total_pheromone

    def pheromone_deposit_function(self, path_cost):
        # Higher deposit for faster paths
        return 1 / path_cost  # Example function, adjust as needed

    def get_simulation_data(self):
        queryTest.query_speed_at_time(self.simulation)
        filename = "query_results-" + str(self.simulation) + ".json"
        with open(filename, 'r') as f:
            data = json.load(f)
        data_dict = {entry['edge_id']: entry for entry in data}
        return data_dict

    def build_graph(self):
        G = nx.DiGraph()
        edges = self.unique_edges

        # Add nodes to the graph
        for edge in edges:
            G.add_node(edge, length=self.edge_map[edge]['length'])

        speed_data = self.get_simulation_data()

        for connection in self.connections:
            entry = speed_data[connection[0]]
            length = self.edge_map[connection[0]]['length']
            max_speed = self.edge_map[connection[0]]['speed_limit']

            bike_speed = max(float(entry['bike_speed']), 0.1)
            car_speed = max(float(entry['car_speed']), 0.1)
            pedestrian_speed = max(float(entry['pedestrian_speed']), 0.1)

            travel_times = {
                'e_bike_1': length / min(bike_speed, max_speed),
                'e_car': length / min(car_speed, max_speed),
                'e_scooter_1': length / min(bike_speed, max_speed),
                'walking': length / min(pedestrian_speed, max_speed)
            }

            # Add edges to the graph with either modified or normal travel times
            G.add_edge(connection[0], connection[1], travel_times=travel_times, distance=length)

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
            if 'e_bike_1' in stations:
                mode_stations['e_bike_1'].append(edge_id)
            if 'e_scooter_1' in stations:
                mode_stations['e_scooter_1'].append(edge_id)
            if 'e_car' in stations:
                mode_stations['e_car'].append(edge_id)
        # print(mode_stations)
        self.mode_stations = mode_stations
        return mode_stations

    def calculate_all_modes_shortest_paths(self):
        flag = True
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
                            total_distance = sum(
                                G[path[i]][path[i + 1]]['distance'] for i in range(len(path) - 1))
                            shortest_paths[(source, target)] = (path, total_time, total_distance)
                        except nx.NetworkXNoPath:
                            print(f"No path found from {source} to {target} for mode {mode}.")
                            flag = False

            all_modes_shortest_paths[mode] = shortest_paths

        return all_modes_shortest_paths, flag

    def calculate_walking_paths_from_start(self, source, destination_edge):
        flag = True
        G = self.G

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
                    total_distance = sum(
                        G[path[i]][path[i + 1]]['distance'] for i in range(len(path) - 1))
                    walking_paths[target] = (path, total_time, total_distance)
                except nx.NodeNotFound:
                    print(f"Node not found from {source} to {target}")
                    flag = False
                except nx.NetworkXNoPath:
                    print(f"No walking path found from {source} to {target}.")
                    flag = False

        return walking_paths, flag

    def calculate_walking_paths_to_destination(self, start_edge, destination_edge):
        flag = True
        G = self.G  # Ensure this graph includes walking times as weights

        # Extract the walking time for an edge
        def walking_time(u, v, d):
            return d['travel_times']['walking']

        # Initialize a structure to hold the shortest paths and their total time costs
        paths_to_destination = {}
        station_classifications = self.classify_stations()
        all_station_edges = set().union(*station_classifications.values(), {start_edge})
        # Calculate paths from all station edges and the start edge to the destination edge using walking mode
        for source in all_station_edges:
            if source != destination_edge:  # Exclude path from the destination to itself
                try:
                    path = nx.shortest_path(G, source=source, target=destination_edge, weight=walking_time)
                    # Calculate the total walking time for the path
                    total_time = sum(G[path[i]][path[i + 1]]['travel_times']['walking'] for i in range(len(path) - 1))
                    total_distance = sum(
                        G[path[i]][path[i + 1]]['distance'] for i in range(len(path) - 1))
                    paths_to_destination[source] = (path, total_time, total_distance)
                except nx.NodeNotFound:
                    print(f"Node not found from {source} to {destination_edge}.")
                    flag = False
                except nx.NetworkXNoPath:
                    print(f"No walking path found from {source} to {destination_edge}.")
                    flag = False
        return paths_to_destination, flag

    def build_new_graph(self, start_edge, destination_edge):
        paths_graph = nx.MultiDiGraph()

        # Collect all station edges
        station_classifications = self.classify_stations()
        all_station_edges = set().union(*station_classifications.values(), {start_edge, destination_edge})
        for node in all_station_edges:
            paths_graph.add_node(node)

        all_modes_shortest_paths, flag = self.calculate_all_modes_shortest_paths()
        if flag is False:
            return None
        for mode, mode_shortest_paths in all_modes_shortest_paths.items():
            for (source, target), (path, total_time, distance) in mode_shortest_paths.items():
                if {source, target}.issubset(all_station_edges):
                    # Add a direct edge with total time cost as the weight
                    paths_graph.add_edge(source, target, weight=total_time, path=path, pheromone_level=0.1,
                                         distance=distance, key=mode)

        walking_paths_from_start, flag = self.calculate_walking_paths_from_start(start_edge, destination_edge)
        if flag is False:
            return None
        walking_paths_to_destination, flag = self.calculate_walking_paths_to_destination(start_edge, destination_edge)
        if flag is False:
            return None
        for target, (path, total_time, distance) in walking_paths_from_start.items():
            if target in all_station_edges:
                paths_graph.add_edge(start_edge, target, weight=total_time, key='walking', path=path,
                                     pheromone_level=0.1, distance=distance)

        for source, (path, total_time, distance) in walking_paths_to_destination.items():
            if source in all_station_edges:
                paths_graph.add_edge(source, destination_edge, weight=total_time, key='walking', path=path,
                                     pheromone_level=0.1, distance=distance)

        self.new_graph = paths_graph
        # print("The graph edges are:", paths_graph.edges)
        # self.visualize_paths_graph(paths_graph)
        return paths_graph

    def visualize_paths_graph(self, paths_graph):
        plt.figure(figsize=(12, 12))  # Increase figure size
        pos = nx.kamada_kawai_layout(paths_graph)  # Use a different layout

        # Draw nodes and edges
        nx.draw_networkx_nodes(paths_graph, pos, node_size=500, node_color='lightblue')
        nx.draw_networkx_edges(paths_graph, pos, edge_color='gray', arrows=True)
        nx.draw_networkx_labels(paths_graph, pos, font_size=8)

        # Prepare edge labels: combine mode and weight
        edge_labels = {(u, v): f"{d['key']} ({d['weight']})" for u, v, d in paths_graph.edges(data=True)}

        # Draw edge labels
        nx.draw_networkx_edge_labels(paths_graph, pos, edge_labels=edge_labels, font_size=7)

        plt.title("Shortest Paths Graph Visualization")
        plt.axis('off')  # Hide axes
        plt.show()

    def pre_computation(self, source, end):
        print("The path between station edges:")
        print(self.calculate_all_modes_shortest_paths())
        print("The walking paths from start to stations and end edge:")
        print(self.calculate_walking_paths_from_start(source, end))
        print("The walking paths from start and stations to end:")
        print(self.calculate_walking_paths_to_destination(source, end))
        self.new_graph = self.build_new_graph(source, end)
        # self.visualize_paths_graph_interactive(new_graph)
        self.visualize_paths_graph(self.new_graph)
        print(self.new_graph.number_of_nodes(), self.new_graph.number_of_edges())

    def update_pheromones(self, ant):
        total_pheromone = 0
        for i in range(0, len(ant.path) - 1):
            edge_1, mode_1, _, _ = ant.path[i]
            edge_2, mode_2, _, _ = ant.path[i + 1]
            # Retrieve all edge data between edge_1 and edge_2
            all_edges_data = self.new_graph.get_edge_data(edge_1, edge_2)

            # Check if there are any edges between these nodes
            if all_edges_data:
                for key, edge_data in all_edges_data.items():
                    if key == mode_2:
                        # Correctly access and update pheromone level
                        current_pheromone_level = edge_data.get('pheromone_level',
                                                                0)  # Get current level, default to 0 if not set
                        # Update pheromone level
                        updated_pheromone_level = current_pheromone_level + self.pheromone_deposit_function(
                            ant.total_time_cost) * 10000
                        # Set the updated pheromone level back on the edge
                        self.new_graph[edge_1][edge_2][key]['pheromone_level'] = updated_pheromone_level

            total_pheromone = total_pheromone + self.new_graph[edge_1][edge_2][key]['pheromone_level']
        return total_pheromone

    def pheromone_deposit_function(self, path_cost):
        # Higher deposit for faster paths
        return 1 / path_cost  # Example function, adjust as needed

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

    def run_aco_algorithm(self, start_edge, destination_edge, number_of_ants, mode='walking'):
        start_time = time.time()
        if start_edge not in self.unique_edges:
            print(f"There's no edge {start_edge} in this map.")
            return
        if destination_edge not in self.unique_edges:
            print(f"There's no edge {destination_edge} in this map.")
            return

        log_interval = 1
        # Initialize ants with specified start and destination locations
        # ants = [Ant(start_edge, destination_edge, self.db_connection, self.edges, mode) for _ in
        #         range(number_of_ants)]

        best_time_cost = float('inf')
        ant_index = 1
        ants = []
        best_path = None
        total_pheromones = {}
        pheromones = []
        max_pheromone = 0
        time_costs = []
        time_cost_counts = {}
        flag = False
        exe_time = 0

        # print(self.edges_station)
        for ant_num in range(number_of_ants):
            ant = Ant(start_edge, destination_edge, self.db_connection, self.mode_stations, ant_index,
                      self.new_graph, mode)
            print("ant", ant_index)
            move_count = 0
            while ant.path[-1][0] != destination_edge and ant.stop is False:
                # and ant.total_time_cost <= best_time_cost
                ant.move()
                if len(ant.path) > 2 and (ant.path[-1][0] == ant.path[-2][0]):
                    break
                move_count += 1
                if move_count > 1000:
                    # print("Move limit reached, breaking out of loop.")
                    break
            # time_costs.append(ant.total_time_cost)
            # print(ant.path)
            if ant.path[-1][0] == destination_edge:
                # paths.append(ant.path)
                # print(ant.path)
                print("time cost:", ant.total_time_cost)

                if ant.total_time_cost <= best_time_cost:
                    # print("curren best time cost", best_time_cost)
                    # print("curren time cost", ant.total_time_cost)
                    time_costs.append(ant.total_time_cost)
                    # print(ant.total_time_cost)
                    ants.append(ant)
                    best_time_cost = ant.total_time_cost
                    total_pheromone = self.update_pheromones(ant)
                    total_pheromones[ant] = total_pheromone
                    print("updated")
                    if len(time_costs) >= 4 and (
                            time_costs[len(time_costs) - 1] == time_costs[len(time_costs) - 2]
                            == time_costs[len(time_costs) - 3] == time_costs[len(time_costs) - 4]):
                        end_time = time.time()
                        exe_time = end_time - start_time
                        flag = True
                # Count the occurrence of the time cost
                if ant.total_time_cost in time_cost_counts:
                    time_cost_counts[ant.total_time_cost] += 1
                else:
                    time_cost_counts[ant.total_time_cost] = 1
            ant_index += 1
        self.reset_graph()
        current_best_ant = self.find_best_path(ants, destination_edge, total_pheromones)  # based on pheromone level
        if current_best_ant is not None:
            best_path = current_best_ant.path
            best_time_cost = current_best_ant.total_time_cost + self.edge_map[destination_edge]['length'] / 1.5
        # best_distance_cost = current_best_ant.path_length

        # if (iteration + 1) % log_interval == 0 or iteration == number_of_iterations - 1: print( f"Iteration {
        # iteration + 1}/{number_of_iterations}: Best Path = {best_path}, Time Cost = {best_time_cost},
        # Total Distance = {best_distance_cost}")

        if not flag:
            end_time = time.time()
            exe_time = end_time - start_time
        # self.visualization(time_costs, number_of_ants)
        return best_path, best_time_cost, exe_time

    def reset_graph(self):
        # Iterate over all edges in the graph
        for u, v, key in self.new_graph.edges(keys=True):
            # Reset the pheromone_level for each edge
            self.new_graph[u][v][key]['pheromone_level'] = 0.1
        print("Graph has been reset with updated pheromone levels.")

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
        plt.title(f"Optimality of Objective Value Over Iterations {ant_num} Ants Used")
        filename = (f"optimization_{ant_num}.png")
        plt.savefig(filename)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Time Cost (seconds) Difference from the Best Time Cost")
        plt.grid(True)
        # plt.show()
