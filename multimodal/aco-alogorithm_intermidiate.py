import random
import matplotlib.pyplot as plt
import networkx as nx

# Constants
STATIONS = ['e-scooter', 'e-bike', 'e-car', 'walking']
ENERGY_MAX = {'e-scooter': 10, 'e-bike': 15, 'e-car': 50, 'walking': float('inf')}
CHANGE_COST = 1  # Cost of changing modes at a location

network = {
    # Each entry: (Next Station, Mode, Energy Cost, Time Cost)
    # Location 'A'
    'A': {
        # 'e-scooter': [('D', 'e-scooter', 2, 10)],
        # 'e-bike': [ ('D', 'e-bike', 3, 15)],
        # 'e-car': [('D', 'e-car', 30, 12)],
        'walking': [('C', 'walking', 0, 30), ('D', 'walking', 0, 60), ('X', 'walking', 0, 10)],
    },

    # Location 'B'
    'B': {
        # 'e-scooter': [('X', 'e-scooter', 2, 7)],
        # 'e-bike': [('D', 'e-bike', 2, 8)],
        # 'e-car': [('C', 'e-car', 20, 6)],
        'walking': [('X', 'walking', 0, 10)],
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

    'X': {
        'walking': [('A', 'walking', 0, 10), ('B', 'walking', 0, 10)]
    }

}


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

    # Pheromone update function
    def update_pheromones(self, ant, evaporation_rate, pheromone_deposit_amount):
        # Loop through the ant's path to update the pheromones
        for i in range(len(ant.path) - 1):
            current_path = (ant.path[i], ant.path[i + 1])
            if current_path in pheromone_levels:
                pheromone_levels[current_path] *= (1 - evaporation_rate)
                pheromone_levels[current_path] += pheromone_deposit_amount

            # Update global pheromone levels
            if current_path in global_pheromone_levels:
                global_pheromone_levels[current_path] *= (1 - evaporation_rate)
                global_pheromone_levels[current_path] += pheromone_deposit_amount
            else:
                global_pheromone_levels[current_path] = pheromone_deposit_amount

    def pheromone_deposit_function(self, path_cost):
        # Higher deposit for shorter paths
        return 1 / path_cost  # Example function, adjust as needed


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

        # Add all next moves from the current location
        for mode in network[current_location]:
            mode_next_moves = network[current_location][mode]
            all_next_moves.extend(mode_next_moves)

        # Check if the only available option is 'walking'
        if len(all_next_moves) == 1 and all_next_moves[0][1] == 'walking':
            self.update_ant_state(all_next_moves[0])
            return

        # Calculate move probabilities and choose the next move
        move_probabilities = self.calculate_move_probabilities(all_next_moves)
        if move_probabilities:
            potential_moves = random.choices(all_next_moves, weights=move_probabilities, k=len(all_next_moves))

            # Check if the path eventually leads to a station for returning the device
            for move in potential_moves:
                if current_mode in ['e-scooter', 'e-bike', 'e-car'] and not self.can_return_device(move[0]):
                    continue  # Skip this move if it doesn't lead to a return station
                self.update_ant_state(move)
                return

            print("No valid path found for moving or returning the device.")

    def can_return_device(self, location):
        # Check if there is a path from the given location to a station where the device can be returned
        if location in network and any(mode in network[location] for mode in ['e-scooter', 'e-bike', 'e-car']):
            return True

        # Check connections from this location to other stations
        for mode in network.get(location, {}):
            for dest, _, _, _ in network[location][mode]:  # Correctly unpack the tuple
                if dest in network and any(mode in network[dest] for mode in ['e-scooter', 'e-bike', 'e-car']):
                    return True

        return False

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

        if current_location not in network:
            print(f"Location {current_location} not found in network")
            return False

        if not any(self.remaining_energy[mode] > 0 for mode in STATIONS):
            print("Energy depleted for all modes")
            return False

        return True


# Function to find the best path
def find_best_path(ants):
    best_ant = None
    best_time_cost = float('inf')
    for ant in ants:
        if ant.total_time_cost < best_time_cost:
            best_time_cost = ant.total_time_cost
            best_ant = ant
    return best_ant


# Training start
# Example start and destination locations
start_location = 'A'
destination_location = 'B'
number_of_ants = 100  # Number of ants to use

# Initialize the network and pheromone levels
network_obj = TransportationNetwork(network, STATIONS, ENERGY_MAX)
pheromone_levels = network_obj.initialize_pheromones()
# Initialize global pheromone levels
global_pheromone_levels = {}

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
        while ant.can_move() and ant.path[-1][0] != 'B':
            ant.move()
            print(
                "-------one move--------")
            move_count += 1
            if move_count > 100:
                print("Move limit reached, breaking out of loop.")
                break

        network_obj.update_pheromones(ant, evaporation_rate,
                                      network_obj.pheromone_deposit_function(ant.total_time_cost))
        print("------one ant-------time cost:", ant.total_time_cost)

    current_best_ant = find_best_path(ants)  # based on time cost
    if current_best_ant and current_best_ant.total_time_cost < best_time_cost:
        best_path = current_best_ant.path
        best_time_cost = current_best_ant.total_time_cost

    ants = [Ant(start_location, destination_location) for _ in range(number_of_ants)]

    if (iteration + 1) % log_interval == 0 or iteration == number_of_iterations - 1:
        print(
            f"Iteration {iteration + 1}/{number_of_iterations}: Best Path = {best_path}, Time Cost = {best_time_cost}")
        print("Global Pheromone Levels:")
        for edge, pheromone_level in global_pheromone_levels.items():
            print(f"{edge}: {pheromone_level}")

print("Best Path after all iterations:", best_path)
