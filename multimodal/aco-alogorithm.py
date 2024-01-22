import random

# Constants
# STATIONS = ['e-scooter', 'e-bike', 'e-car', 'walking']
# ENERGY_MAX = {'e-scooter': 10, 'e-bike': 15, 'e-car': 50, 'walking': float('inf')}
# CHANGE_COST = 1  # Cost of changing modes at a location
#
# # Representation of the network
# network = {
#     # Each entry: (Next Station, Energy Cost, Time Cost)
#     # Paths from location 'A'
#     ('A', 'e-scooter'): [('B', 'e-scooter', 1, 5), ('C', 'e-scooter', 2, 10)],
#     ('A', 'e-bike'): [('B', 'e-bike', 1, 4), ('D', 'e-bike', 3, 15)],
#     ('A', 'e-car'): [('C', 'e-car', 20, 8), ('D', 'e-car', 30, 12)],
#
#     # Paths from location 'B'
#     ('B', 'e-scooter'): [('C', 'e-scooter', 2, 7), ('D', 'e-scooter', 3, 10)],
#     ('B', 'e-bike'): [('A', 'e-bike', 1, 4), ('D', 'e-bike', 2, 8)],
#     ('B', 'e-car'): [('A', 'e-car', 10, 11), ('C', 'e-car', 20, 6)],
#
#     # Paths from location 'C'
#     ('C', 'e-scooter'): [('A', 'e-scooter', 2, 10), ('D', 'e-scooter', 3, 9)],
#     ('C', 'e-bike'): [('B', 'e-bike', 2, 7), ('D', 'e-bike', 2, 6)],
#     ('C', 'e-car'): [('A', 'e-car', 20, 12), ('B', 'e-car', 10, 5)],
#
#     # Paths from location 'D'
#     ('D', 'e-scooter'): [('A', 'e-scooter', 4, 14), ('B', 'e-scooter', 3, 10)],
#     ('D', 'e-bike'): [('C', 'e-bike', 2, 6), ('A', 'e-bike', 3, 15)],
#     ('D', 'e-car'): [('B', 'e-car', 10, 7), ('C', 'e-car', 30, 8)],

# # Walking paths within locations on changing
# ('A', 'walking'): [('A', 'e-scooter', CHANGE_COST, 1), ('A', 'e-bike', CHANGE_COST, 1),
#                    ('A', 'e-car', CHANGE_COST, 1)],
# ('B', 'walking'): [('B', 'e-scooter', CHANGE_COST, 1), ('B', 'e-bike', CHANGE_COST, 1),
#                    ('B', 'e-car', CHANGE_COST, 1)],
# ('C', 'walking'): [('C', 'e-scooter', CHANGE_COST, 1), ('C', 'e-bike', CHANGE_COST, 1),
#                    ('C', 'e-car', CHANGE_COST, 1)],
# ('D', 'walking'): [('D', 'e-scooter', CHANGE_COST, 1), ('D', 'e-bike', CHANGE_COST, 1),
#                    ('D', 'e-car', CHANGE_COST, 1)],
#
# # Walking paths to different locations
# Each entry: (Next Edge, Time Cost)
#     ('A', 'walking'): [('B', 'walking', 0, 20), ('C', 'walking', 0, 30), ('D', 'walking', 0, 60)],
#     ('B', 'walking'): [('A', 'walking', 0, 20), ('C', 'walking', 0, 30), ('D', 'walking', 0, 20)],
#     ('C', 'walking'): [('A', 'walking', 0, 30), ('B', 'walking', 0, 30), ('D', 'walking', 0, 20)],
#     ('D', 'walking'): [('A', 'walking', 0, 60), ('B', 'walking', 0, 20), ('C', 'walking', 0, 20)]
# }


# class TransportationNetwork:
#     def __init__(self, locations, stations, energy_constraints):
#         self.network = network  # Location connectivity and time costs
#         self.stations = STATIONS  # Stations available at each location
#         self.energy_max = ENERGY_MAX  # Max energy available for each mode
#         # self.heuristic_info = self.calculate_heuristic_info()
#         # self.pheromone_levels = self.initialize_pheromones()
#
#     def initialize_pheromones(self):
#         # Pheromone levels
#         pheromone_levels = {}
#         # Initialize pheromone levels for all possible paths
#         for start_loc in ['A', 'B', 'C', 'D']:
#             for start_mode in STATIONS:
#                 for end_loc in ['A', 'B', 'C', 'D']:
#                     for end_mode in STATIONS:
#                         pheromone_levels[((start_loc, start_mode), (end_loc, end_mode))] = 0.1
#
#         return pheromone_levels
#
#     def calculate_heuristic_info(self):
#         # Calculate heuristic information for each path
#         heuristic_info = {}
#         for key in self.network:
#             heuristic_info[key] = []
#             for _, _, energy_cost, time_cost in self.network[key]:
#                 heuristic_info[key].append(1 / time_cost)  # or any other heuristic measure
#         return heuristic_info
#
#
# # Ant class definition
# class Ant:
#     def __init__(self, start_location, destination_location):
#         self.start_location = start_location
#         self.destination_location = destination_location
#         self.path = [(start_location, 'e-scooter')]  # Initialize with starting location and mode
#         self.remaining_energy = ENERGY_MAX.copy()
#         self.total_time_cost = 0
#
#     def move(self):
#         print("move")
#         current_station = self.path[-1]
#         print(f"Current station: {current_station}")
#         # collection of next move
#         next_moves = network[current_station]
#         print(f"Next possible moves: {next_moves}")
#         # Filter out moves that lead to immediate cycles
#         next_moves = [move for move in next_moves if not self.is_cycle(move)]
#
#         move_probabilities = self.calculate_move_probabilities(next_moves)
#         if move_probabilities:
#             next_station = random.choices(next_moves, weights=move_probabilities, k=1)[0]
#             self.update_ant_state(next_station)
#
#     def is_cycle(self, next_move):
#         # Check if the next move leads to a location that has been recently visited
#         if len(self.path) > 2 and next_move[0] == self.path[-2][0]:
#             return True
#         return False
#
#     def calculate_move_probabilities(self, next_moves):
#         alpha = 1  # Pheromone importance
#         beta = 2  # Heuristic importance
#         gamma = 3  # Energy importance
#         probabilities = []
#         current_loc, current_mode = self.path[-1]  # Current location and mode
#
#         for next_move in next_moves:
#             # Unpacking the move details
#             if len(next_move) == 4:
#                 dest, mode, energy_cost, time_cost = next_move
#             elif len(next_move) == 3:
#                 dest, mode, time_cost = next_move
#                 energy_cost = 0
#             else:
#                 continue  # Skip if the tuple structure is unexpected
#
#             pheromone_key = ((current_loc, current_mode), (dest, mode))
#             pheromone_level = pheromone_levels.get(pheromone_key, 0.1)
#             heuristic = 1 / time_cost
#             energy_factor = self.remaining_energy[mode] / ENERGY_MAX[mode]
#
#             probability = (pheromone_level ** alpha) * (heuristic ** beta) * (energy_factor ** gamma)
#             probabilities.append(probability)
#
#         total = sum(probabilities)
#         return [p / total for p in probabilities] if total > 0 else []
#
#     def update_ant_state(self, next_station):
#         # Check the length of next_station to unpack correctly
#         if len(next_station) == 4:
#             dest, mode, energy_cost, time_cost = next_station
#             print(f"Moving to {dest} using {mode}")  # Print destination and mode
#             print(f"Energy cost: {energy_cost}, Time cost: {time_cost}")  # Print energy and time costs
#         elif len(next_station) == 3:
#             dest, mode, time_cost = next_station
#             energy_cost = 0  # For walking, there is no energy cost
#         else:
#             return  # Skip if the tuple structure is unexpected
#
#         # Update path, energy, and time cost
#         self.path.append((dest, mode))
#         self.remaining_energy[mode] -= energy_cost
#         self.total_time_cost += time_cost
#
#     def can_move(self):
#         current_location, current_mode = self.path[-1]
#         print(f"Current location: {current_location}, Current mode: {current_mode}")
#
#         network_key = (current_location, current_mode)
#         if network_key not in network:
#             print(f"Location {network_key} not found in network")
#             return False
#
#         if not any(self.remaining_energy[mode] > 0 for mode in STATIONS):
#             print("Energy depleted for all modes")
#             return False
#
#         return True
#
#
# # Pheromone update function
# def update_pheromones(ant, evaporation_rate, pheromone_deposit_amount):
#     # Loop through the ant's path to update the pheromones
#     for i in range(len(ant.path) - 1):
#         current_path = (ant.path[i], ant.path[i + 1])
#         if current_path in pheromone_levels:
#             pheromone_levels[current_path] *= (1 - evaporation_rate)
#             pheromone_levels[current_path] += pheromone_deposit_amount
#
#
# def pheromone_deposit_function(path_cost):
#     # Higher deposit for shorter paths
#     return 1 / path_cost  # Example function, adjust as needed
#
#
# # Function to find the best path
# def find_best_path(ants):
#     best_ant = None
#     best_time_cost = float('inf')
#     for ant in ants:
#         if ant.total_time_cost < best_time_cost:
#             best_time_cost = ant.total_time_cost
#             best_ant = ant
#     return best_ant
#
#
# # Example start and destination locations
# start_location = 'A'
# destination_location = 'D'
# number_of_ants = 10  # Number of ants to use
#
# # Initialize the network and pheromone levels
# network_obj = TransportationNetwork(['A', 'B', 'C', 'D'], STATIONS, ENERGY_MAX)
# pheromone_levels = network_obj.initialize_pheromones()
#
# # Run the ACO algorithm
# number_of_iterations = 10
# evaporation_rate = 0.1
# pheromone_deposit_amount = 1
# log_interval = 1
#
# # Initialize ants with specified start and destination locations
# ants = [Ant(start_location, destination_location) for _ in range(number_of_ants)]
#
# best_path = None
# best_time_cost = float('inf')
#
# for iteration in range(number_of_iterations):
#     print(f"Starting iteration {iteration + 1}")
#     for ant in ants:
#         move_count = 0
#         while ant.can_move() and ant.path[-1][0] != 'D':
#             ant.move()
#             move_count += 1
#             if move_count > 100:
#                 print("Move limit reached, breaking out of loop.")
#                 break
#         update_pheromones(ant, evaporation_rate, pheromone_deposit_amount)
#
#     current_best_ant = find_best_path(ants)
#     if current_best_ant and current_best_ant.total_time_cost < best_time_cost:
#         best_path = current_best_ant.path
#         best_time_cost = current_best_ant.total_time_cost
#
#     ants = [Ant(start_location, destination_location) for _ in range(number_of_ants)]
#
# if (iteration + 1) % log_interval == 0 or iteration == number_of_iterations - 1: print(f"Iteration {iteration +
# 1}/{number_of_iterations}: Best Path = {best_path}, Time Cost = {best_time_cost}")
#
# print("Best Path after all iterations:", best_path)


# Constants
STATIONS = ['e-scooter', 'e-bike', 'e-car', 'walking']
ENERGY_MAX = {'e-scooter': 10, 'e-bike': 15, 'e-car': 50, 'walking': float('inf')}
CHANGE_COST = 1  # Cost of changing modes at a location

network = {
    # Location 'A'
    'A': {
        'e-scooter': [('B', 'e-scooter', 1, 5), ('C', 'e-scooter', 2, 10)],
        'e-bike': [('B', 'e-bike', 1, 4), ('D', 'e-bike', 3, 15)],
        'e-car': [('C', 'e-car', 20, 8), ('D', 'e-car', 30, 12)],
        'walking': [('B', 'walking', 0, 20), ('C', 'walking', 0, 30), ('D', 'walking', 0, 60)],
    },

    # Location 'B'
    'B': {
        'e-scooter': [('C', 'e-scooter', 2, 7), ('D', 'e-scooter', 3, 10)],
        'e-bike': [('A', 'e-bike', 1, 4), ('D', 'e-bike', 2, 8)],
        'e-car': [('A', 'e-car', 10, 11), ('C', 'e-car', 20, 6)],
        'walking': [('A', 'walking', 0, 20), ('C', 'walking', 0, 30), ('D', 'walking', 0, 20)],
    },

    # Location 'C'
    'C': {
        'e-scooter': [('A', 'e-scooter', 2, 10), ('D', 'e-scooter', 3, 9)],
        'e-bike': [('B', 'e-bike', 2, 7), ('D', 'e-bike', 2, 6)],
        'e-car': [('A', 'e-car', 20, 12), ('B', 'e-car', 10, 5)],
        'walking': [('A', 'walking', 0, 30), ('B', 'walking', 0, 30), ('D', 'walking', 0, 20)],
    },

    # Location 'D'
    'D': {
        'e-scooter': [('A', 'e-scooter', 4, 14), ('B', 'e-scooter', 3, 10)],
        'e-bike': [('C', 'e-bike', 2, 6), ('A', 'e-bike', 3, 15)],
        'e-car': [('B', 'e-car', 10, 7), ('C', 'e-car', 30, 8)],
        'walking': [('A', 'walking', 0, 60), ('B', 'walking', 0, 20), ('C', 'walking', 0, 20)],
    },
}

import matplotlib.pyplot as plt
import networkx as nx

# Constants and Network definition as provided before...

# Create a MultiDiGraph to support multiple edges between the same nodes
G = nx.MultiDiGraph()

# Add nodes for each location
for loc in network.keys():
    G.add_node(loc)

# Add edges with labels for all transportation modes
for loc in network:
    for mode in STATIONS:
        for dest, dest_mode, energy_cost, time_cost in network[loc][mode]:
            # Add edges only between the same transportation modes
            if mode == dest_mode:
                G.add_edge(f"{loc}-{mode}", f"{dest}-{dest_mode}", weight=time_cost,
                           label=f"{mode}\nEnergy: {energy_cost}, Time: {time_cost}")

# Position the nodes in a circular layout and then offset station nodes
loc_pos = nx.circular_layout([loc for loc in network])  # Position for main locations
pos = {}  # Dictionary to hold the positions of all nodes

plt.figure(figsize=(20, 20))

# Offset for station nodes to avoid overlap
offset = {mode: (idx * 0.1, idx * 0.1) for idx, mode in enumerate(STATIONS)}

# Assign positions to location nodes and offset station nodes
for loc in network:
    pos[loc] = loc_pos[loc]
    for mode in STATIONS:
        pos[f"{loc}-{mode}"] = loc_pos[loc] + offset[mode]

# Draw the nodes
nx.draw_networkx_nodes(G, pos, node_size=300, node_color="lightblue")

# Draw the edges and labels
for u, v, data in G.edges(data=True):
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=1)
    label = data['label']
    x, y = (pos[u][0] * 0.6 + pos[v][0] * 0.4, pos[u][1] * 0.6 + pos[v][1] * 0.4)
    plt.text(x, y, label, size=8, color='black', ha='center', va='center')

# Draw location labels
nx.draw_networkx_labels(G, pos, labels={loc: loc for loc in network}, font_size=10)

plt.axis('off')
plt.show()


class TransportationNetwork:
    def __init__(self, network, stations, energy_constraints):
        self.network = network  # Location connectivity and time costs
        self.stations = stations  # Stations available at each location
        self.energy_max = energy_constraints  # Max energy available for each mode

    def initialize_pheromones(self):
        # Pheromone levels
        pheromone_levels = {}
        # Initialize pheromone levels for all possible paths
        for start_loc in network.keys():
            for start_mode in STATIONS:
                for end_loc in network.keys():
                    for end_mode in STATIONS:
                        pheromone_levels[((start_loc, start_mode), (end_loc, end_mode))] = 0.1

        return pheromone_levels

    def calculate_heuristic_info(self):
        # Calculate heuristic information for each path
        heuristic_info = {}
        for key in self.network:
            heuristic_info[key] = []
            for _, _, energy_cost, time_cost in self.network[key][STATIONS[0]]:
                heuristic_info[key].append(1 / time_cost)  # or any other heuristic measure
        return heuristic_info


# Ant class definition
class Ant:
    def __init__(self, start_location, destination_location):
        self.start_location = start_location
        self.destination_location = destination_location
        self.path = [(start_location, 'e-scooter')]  # Initialize with starting location and mode
        self.remaining_energy = ENERGY_MAX.copy()
        self.total_time_cost = 0
        self.device_to_return = True  # Flag to track if a device needs to be returned

    def return_device(self):
        current_location, current_mode = self.path[-1]

        # Check if the current mode is a device that needs to be returned
        if current_mode in ['e-scooter', 'e-bike', 'e-car']:
            # Replenish the energy for the returned mode to its initial value
            self.remaining_energy[current_mode] = ENERGY_MAX[current_mode]

    def move(self):
        current_location, current_mode = self.path[-1]

        # Initialize a list to collect all possible moves
        all_next_moves = []

        # Check if the ant is using a device that needs to be returned
        if self.device_to_return:
            # Limit the next moves to the same mode to return the device
            next_moves = network[current_location][current_mode]
            all_next_moves.extend(next_moves)
            print("all next moves for return stations", all_next_moves)
            # fulfill the energy
            if len(self.path) >= 2 and self.path[-2][1] != current_mode:
                self.return_device()
        else:
            # If no device needs to be returned, consider all modes
            for mode in STATIONS:
                mode_next_moves = network[current_location][mode]
                all_next_moves.extend(mode_next_moves)
                print("all next moves", all_next_moves)

        # Calculate move probabilities and choose the next move
        move_probabilities = self.calculate_move_probabilities(all_next_moves)
        print("move probabilities:", move_probabilities)
        if move_probabilities:
            next_station = random.choices(all_next_moves, weights=move_probabilities, k=1)[0]
            self.update_ant_state(next_station)

    def is_cycle(self, next_move):
        # Check if the next move leads to a location that has been recently visited
        if len(self.path) > 2 and next_move[0] == self.path[-2][0]:
            return True
        return False

    def calculate_move_probabilities(self, next_moves):
        alpha = 1  # Pheromone importance
        beta = 2  # Heuristic importance
        gamma = 3  # Energy importance
        probabilities = []
        current_loc, current_mode = self.path[-1]  # Current location and mode

        for next_move in next_moves:
            dest, mode, energy_cost, time_cost = next_move

            pheromone_key = ((current_loc, current_mode), (dest, mode))
            pheromone_level = pheromone_levels.get(pheromone_key, 0.1)
            heuristic = 1 / time_cost if time_cost > 0 else 0  # Avoid division by zero
            # Special handling for 'walking' mode
            if mode == 'walking':
                energy_factor = 1.0
            else:
                energy_factor = self.remaining_energy[mode] / ENERGY_MAX[mode] if ENERGY_MAX[mode] > 0 else 0

            # Debugging print statements
            print(
                f"Move: {next_move}, Pheromone Level: {pheromone_level}, Heuristic: {heuristic}, Energy Factor: {energy_factor}")

            probability = (pheromone_level ** alpha) * (heuristic ** beta) * (energy_factor ** gamma)
            probabilities.append(probability)

        total = sum(probabilities)
        print(f"Total Probability: {total}")  # Debugging print statement
        return [p / total for p in probabilities] if total > 0 else []

    def update_ant_state(self, next_station):
        dest, mode, energy_cost, time_cost = next_station

        # Update the device_to_return flag
        if mode in ['e-scooter', 'e-bike', 'e-car']:
            # If the current mode is the same, it means the device is being returned
            if self.path[-1][1] == mode:
                self.device_to_return = False
            else:
                self.device_to_return = True
        else:
            self.device_to_return = False

        # Update path, energy, and time cost
        self.path.append((dest, mode))
        self.remaining_energy[mode] -= energy_cost
        self.total_time_cost += time_cost

    def can_move(self):
        current_location, current_mode = self.path[-1]
        print(f"Current location: {current_location}, Current mode: {current_mode}")

        if current_location not in network:
            print(f"Location {current_location} not found in network")
            return False

        if current_mode not in network[current_location]:
            print(f"Mode {current_mode} not found in network for location {current_location}")
            return False

        if not any(self.remaining_energy[mode] > 0 for mode in STATIONS):
            print("Energy depleted for all modes")
            return False

        return True


# Pheromone update function
def update_pheromones(ant, evaporation_rate, pheromone_deposit_amount):
    # Loop through the ant's path to update the pheromones
    for i in range(len(ant.path) - 1):
        current_path = (ant.path[i], ant.path[i + 1])
        if current_path in pheromone_levels:
            pheromone_levels[current_path] *= (1 - evaporation_rate)
            pheromone_levels[current_path] += pheromone_deposit_amount


def pheromone_deposit_function(path_cost):
    # Higher deposit for shorter paths
    return 1 / path_cost  # Example function, adjust as needed


# Function to find the best path
def find_best_path(ants):
    best_ant = None
    best_time_cost = float('inf')
    for ant in ants:
        if ant.total_time_cost < best_time_cost:
            best_time_cost = ant.total_time_cost
            best_ant = ant
    return best_ant


# Example start and destination locations
start_location = 'A'
destination_location = 'D'
number_of_ants = 100  # Number of ants to use

# Initialize the network and pheromone levels
network_obj = TransportationNetwork(network, STATIONS, ENERGY_MAX)
pheromone_levels = network_obj.initialize_pheromones()

# Run the ACO algorithm
number_of_iterations = 10
evaporation_rate = 0.1
pheromone_deposit_amount = 0
log_interval = 1

# Initialize ants with specified start and destination locations
ants = [Ant(start_location, destination_location) for _ in range(number_of_ants)]

best_path = None
best_time_cost = float('inf')

for iteration in range(number_of_iterations):
    print(f"Starting iteration {iteration + 1}")
    for ant in ants:
        move_count = 0
        while ant.can_move() and ant.path[-1][0] != 'D':
            ant.move()
            print(
                "-------one move--------")
            move_count += 1
            if move_count > 100:
                print("Move limit reached, breaking out of loop.")
                break

        update_pheromones(ant, evaporation_rate, pheromone_deposit_function(ant.total_time_cost))
        print("------one ant-------time cost:", ant.total_time_cost)

    current_best_ant = find_best_path(ants)  # based on time cost
    if current_best_ant and current_best_ant.total_time_cost < best_time_cost:
        best_path = current_best_ant.path
        best_time_cost = current_best_ant.total_time_cost

    ants = [Ant(start_location, destination_location) for _ in range(number_of_ants)]

    if (iteration + 1) % log_interval == 0 or iteration == number_of_iterations - 1:
        print(
            f"Iteration {iteration + 1}/{number_of_iterations}: Best Path = {best_path}, Time Cost = {best_time_cost}")

print("Best Path after all iterations:", best_path)
