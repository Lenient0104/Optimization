import sqlite3
import time
from aco import Ant
import xml.etree.ElementTree as ET
from tqdm import tqdm
from e_car import ECar_EnergyConsumptionModel
from e_bike import Ebike_PowerConsumptionCalculator
from e_scooter import Escooter_PowerConsumptionCalculator


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

    # def convert_node_path_to_edge_path(self, node_path):
    #     edge_path = []
    #     for i in range(len(node_path) - 1):
    #         edge_id = self.edge_mapping.get((node_path[i], node_path[i + 1]))
    #         if edge_id is None:
    #             raise ValueError(f"No edge found for nodes {node_path[i]} -> {node_path[i + 1]}")
    #         edge_path.append(edge_id)
    #     return edge_path

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
                from_ebike_speed, user_weight, 1, user_pal) * edge_data['from_length']
            edge_data['energy_consumption']['e_bike_1']['to_energy'] = self.ebike_energymodel.calculate(to_ebike_speed,
                                                                                                        user_weight, 1,
                                                                                                        user_pal) * \
                                                                       edge_data['to_length']

            edge_data['energy_consumption']['e_car']['from_energy'] = self.ecar_energymodel.calculate_energy_loss(
                from_ecar_speed) * edge_data['from_length']
            edge_data['energy_consumption']['e_car']['to_energy'] = self.ecar_energymodel.calculate_energy_loss(
                to_ecar_speed) * edge_data['to_length']

            # edge_data['energy_consumption']['e_car']['from_energy'] = 1
            # edge_data['energy_consumption']['e_car']['to_energy'] = 1

            edge_data['energy_consumption']['e_scooter_1']['from_energy'] = self.escooter_energymodel.calculate(
                from_escooter_speed, user_weight, 1) * edge_data['from_length']
            edge_data['energy_consumption']['e_scooter_1']['to_energy'] = self.escooter_energymodel.calculate(
                to_escooter_speed, user_weight, 1) * edge_data['to_length']

    def map_station_availability(self):
        # static information
        cursor = self.db_connection.cursor()
        query = "SELECT StationEdgeID, StationType FROM StationLocation"
        cursor.execute(query)
        station_data = cursor.fetchall()
        print(station_data)
        # Reset station availability in edges
        # for connection_id in self.edges:
        # self.edges[connection_id]['station_availability']['from'] = {'e_bike_1': False, 'e_car': False,
        #                                                              'e_scooter_1': False, 'walking': True}
        # self.edges[connection_id]['station_availability']['to'] = {'e_bike_1': False, 'e_car': False,
        #                                                          'e_scooter_1': False, 'walking': True}

        # Update station availability based on database data
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
    def find_best_path(self, ants, destination_edge):
        best_ant = None
        best_time_cost = float('inf')

        for ant in ants:
            # Check if the ant has reached the destination
            if ant.path[-1][0] == destination_edge:
                # If the ant has reached the destination and has a lower total time cost
                if ant.total_time_cost < best_time_cost:
                    best_time_cost = ant.total_time_cost
                    best_ant = ant

        return best_ant

    def run_aco_algorithm(self, start_edge, destination_edge, number_of_ants, number_of_iterations):
        log_interval = 1
        # Initialize ants with specified start and destination locations
        ants = [Ant(start_edge, destination_edge, 'e_car', self.db_connection, self.edges) for _ in
                range(number_of_ants)]
        best_path = None
        best_time_cost = float('inf')

        for iteration in range(number_of_iterations):
            print(f"Starting iteration {iteration + 1}")
            for ant in ants:
                move_count = 0
                while ant.path[-1][0] != destination_edge:
                    ant.move()
                    if len(ant.path) > 2 and (ant.path[-1][0] == ant.path[-3][0]):
                        break
                    move_count += 1
                    if move_count > 100:
                        print("Move limit reached, breaking out of loop.")
                        break

                print("time cost:", ant.total_time_cost)
                print("----------for this ant the path is-----------", ant.path)

            current_best_ant = self.find_best_path(ants, destination_edge)  # based on time cost
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

    # def map_speeds_to_lanes(self, edges):
    #     cursor = self.db_connection.cursor()
    #     query = """
    #             SELECT EdgeID, Speed
    #             FROM EdgeSpeed
    #             WHERE TimeStep = (SELECT MAX(TimeStep) FROM EdgeSpeed)
    #             """
    #     cursor.execute(query)
    #     latest_speed_data = cursor.fetchall()
    #     # Create a dictionary for fast lookup of mean_speed based on edge_id
    #     speed_lookup = {edge_id: speed for edge_id, speed in latest_speed_data}
    #
    #     # Update the speed in the edges dictionary
    #     for connection_id, edge_data in self.edges.items():
    #         from_edge = edge_data['from_edge']
    #         to_edge = edge_data['to_edge']
    #
    #         if from_edge in speed_lookup:
    #             edge_data['from_speed'] = speed_lookup[from_edge]
    #             edges[connection_id]['from_speed'] = speed_lookup[from_edge]
    #
    #         if to_edge in speed_lookup:
    #             edge_data['to_speed'] = speed_lookup[to_edge]
    #             edges[connection_id]['to_speed'] = speed_lookup[to_edge]
    #
    #         # Calculate time based on updated speeds
    #         from_speed = edge_data.get('from_speed', edge_data['speed_limit'])
    #         to_speed = edge_data.get('to_speed', edge_data['speed_limit'])
    #         from_length = edge_data['from_length']
    #         to_length = edge_data['to_length']
    #
    #         time_from = from_length / from_speed if from_speed != 0 else float('inf')
    #         time_to = to_length / to_speed if to_speed != 0 else float('inf')
    #
    #         edge_data['time'] = time_from + time_to
    #         edges[connection_id]['time'] = time_from + time_to

    # def map_energy_to_lanes(self, user_weight, user_pal, edges): for connection_id, edge_data in tqdm(edges.items(
    # ), desc="Processing edges"): from_speed = edges[connection_id]['from_speed'] to_speed = edges[connection_id][
    # 'to_speed'] edges[connection_id]['from_ebike_energy'] = self.ebike_energymodel.calculate(from_speed,
    # user_weight, 0, user_pal) edges[connection_id]['to_ebike_energy'] = self.ebike_energymodel.calculate(to_speed,
    # user_weight, 0, user_pal)
    #
    #         edges[connection_id]['from_ecar_energy'] = self.ecar_energymodel.calculate_energy_loss(from_speed)
    #         edges[connection_id]['to_ecar_energy'] = self.ecar_energymodel.calculate_energy_loss(to_speed)
    #
    # edges[connection_id]['from_escooter_energy'] = self.escooter_energymodel.calculate(from_speed, user_weight,
    # 0) edges[connection_id]['to_escooter_energy'] = self.escooter_energymodel.calculate(to_speed, user_weight, 0)

    # def build_graph_from_edges(self, edges):
    #     # Initialize a directed graph
    #     G = nx.DiGraph()
    #
    #     # Prepare edge data for batch insertion
    #     edge_data = []
    #     for connection_id, data in tqdm(edges.items(), desc="Adding edges to graph"):
    #         # Create a single tuple for each edge (start, end, attr_dict)
    #         attr_dict = {
    #             'weight': data.get('time', float('inf')),
    #             'connection_id': connection_id,
    #             'length': data['length']
    #         }
    #
    #         if 'from_speed' in data and 'to_speed' in data:
    #             attr_dict['from_speed'] = data['from_speed']
    #             attr_dict['to_speed'] = data['to_speed']
    #
    #         edge_data.append((data['from_edge'], data['to_edge'], attr_dict))
    #
    #     # Add all edges to the graph in a batch
    #     G.add_edges_from(edge_data)
    #
    #     return G
    #
    # def constrained_shortest_path(self, source_edge, target_edge, max_time, new_df):
    #     self.update_edge_speed(new_df)
    #     # Find the shortest path with a constraint on maximum time
    #     path = self.dijkstra_with_max_time(self.G, source_edge, target_edge, max_time)
    #     # path.insert(0, self.edges[source_edge]['from'])
    #     # path.append(self.edges[target_edge]['to'])
    #     print(f"Path from dijkstra_with_max_time: {path}")  # Debug line
    #     if not path:
    #         path = nx.shortest_path(self.G, source_edge, target_edge, weight='weight')
    #         total_time = sum(self.G[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
    #         print(f"Warning: The shortest path takes {total_time:.2f} which exceeds the limit of {max_time}.")
    #         return path
    #     total_time = sum(self.G[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
    #     print(f"The shortest path within the limit takes {total_time:.2f}.")
    #     return path
    #
    # def dijkstra_with_max_time(self, G, source, target, max_time):
    #     # Dijkstra's algorithm modified with a maximum time constraint
    #     dist = {node: float('infinity') for node in G}
    #     dist[source] = 0
    #     pred = {node: None for node in G}
    #     to_explore = [(0, source)]
    #
    #     print("Starting Dijkstra's with max time...")
    #     print(f"Source: {source}, Target: {target}, Max Time: {max_time}")
    #
    #     # Main loop for Dijkstra's algorithm
    #     while to_explore:
    #         accumulated_time, current = min(to_explore)
    #         to_explore.remove((accumulated_time, current))
    #
    #         # print(f"Exploring node {current} with distance {d}")
    #
    #         if current == target and accumulated_time <= max_time:
    #             print(f"Found target {target} with distance {accumulated_time}")
    #             # Construct the path from source to target
    #             path = []
    #             while current:
    #                 # print(f"Adding {current} to the path")  # Add this debug line
    #                 path.append(current)
    #                 current = pred[current]
    #                 if current == source:  # Add this check
    #                     print("Reached the source node in path reconstruction")
    #             print("Constructed path:", path)  # Add this debug line
    #             return path[::-1]
    #
    #         for neighbor in G[current]:
    #             new_dist = accumulated_time + G[current][neighbor]['weight']
    #             if new_dist < dist[neighbor] and new_dist <= max_time:
    #                 dist[neighbor] = new_dist
    #                 pred[neighbor] = current
    #                 to_explore.append((new_dist, neighbor))
    #                 # print(f"Updated distance for node {neighbor}: {new_dist}")
    #             elif new_dist > max_time:
    #                 print(f"Node {neighbor}'s distance {new_dist} exceeds max_time")
    #
    #     print("Couldn't find a valid path within max_time.")
    #     return None
    #
    # def dijkstra_top_three_routes(self, source, target, max_time, G):
    #     # Find top 3 shortest paths using Dijkstra's algorithm with a maximum time constraint
    #     dist = {node: float('infinity') for node in G}
    #     dist[source] = 0
    #     pred = {node: None for node in G}
    #     to_explore = [(0, source)]
    #     found_paths = []
    #
    #     while to_explore and len(found_paths) < 3:
    #         d, current = min(to_explore)
    #         to_explore.remove((d, current))
    #
    #         if current == target and d <= max_time:
    #             path = []
    #             while current:
    #                 path.append(current)
    #                 current = pred[current]
    #             found_paths.append((d, path[::-1]))
    #             continue
    #
    #         for neighbor in G[current]:
    #             new_dist = d + G[current][neighbor]['weight']
    #             if new_dist < dist[neighbor] and new_dist <= max_time:
    #                 dist[neighbor] = new_dist
    #                 pred[neighbor] = current
    #                 to_explore.append((new_dist, neighbor))
    #
    #     found_paths.sort(key=lambda x: x[0])
    #     return [route for _, route in found_paths]
    #
    # def plot_graph_with_path(self, path, G):
    #     # Plot the graph with a particular path highlighted
    #     plt.figure(figsize=(50, 50))
    #     pos = nx.spring_layout(G)
    #     nx.draw_networkx_nodes(G, pos)
    #     nx.draw_networkx_labels(G, pos)
    #     nx.draw_networkx_edges(G, pos, edge_color='grey')
    #
    #     if path:
    #         edges_in_path = list(zip(path[:-1], path[1:]))
    #         nx.draw_networkx_edges(G, pos, edgelist=edges_in_path, edge_color='r', width=2)
    #     plt.show()
    #
    # def analyze_route(self, source_edge, target_edge, max_time_limit):
    #     # Get the 'to' node of the source edge and the 'from' node of the target edge
    #     source_node = self.edges[source_edge]['to']
    #     target_node = self.edges[target_edge]['from']
    #
    #     # Analyze the best route from source to target with a maximum time constraint
    #     path = self.constrained_shortest_path(source_node, target_node, max_time_limit)
    #     if path:
    #         path.insert(0, self.edges[source_edge]['from'])
    #         path.append(target_node)
    #     self.plot_graph_with_path(path)
    #     return self.dijkstra_top_three_routes(source_node, target_node, max_time_limit)
    #
    # # def csv2df(self, csv_path):
    # #     # Read the new CSV file
    # #     df = pd.read_csv(csv_path)
    # #     return df
    #
    # def update_edge_speed(self, df):
    #
    #     print(df)
    #
    #     # Create a dictionary for fast lookup of mean_speed based on edge_id
    #     speed_lookup = df.set_index('edge_id')['mean_speed'].to_dict()
    #
    #     updated_edges = set()
    #
    #     for connection_id, edge_data in tqdm(self.edges.items(), desc="Updating edge speeds"):
    #         from_edge = edge_data['from_edge']
    #         to_edge = edge_data['to_edge']
    #
    #         if from_edge in speed_lookup:
    #             self.edges[connection_id]['from_speed'] = speed_lookup[from_edge]
    #             updated_edges.add(from_edge)
    #
    #         if to_edge in speed_lookup:
    #             self.edges[connection_id]['to_speed'] = speed_lookup[to_edge]
    #             updated_edges.add(to_edge)
    #
    #         # If both the from_edge and to_edge have their speed updated, compute the time
    #         if from_edge in updated_edges and to_edge in updated_edges:
    #             from_speed = self.edges[connection_id].get('from_speed', edge_data['speed_limit'])
    #             to_speed = self.edges[connection_id].get('to_speed', edge_data['speed_limit'])
    #
    #             # The time is a weighted average based on the lengths and speeds of the individual edges
    #             from_length = edge_data['from_length']
    #             to_length = edge_data['to_length']
    #
    #             time_from = from_length / from_speed if from_speed != 0 else float('inf')
    #             time_to = to_length / to_speed if to_speed != 0 else float('inf')
    #
    #             self.edges[connection_id]['time'] = time_from + time_to
    #
    #         # Update the graph (NetworkX) to reflect this change
    #         if (edge_data['from_edge'], edge_data['to_edge']) in self.G.edges:
    #             self.G[edge_data['from_edge']][edge_data['to_edge']]['weight'] = self.edges[connection_id]['time']
