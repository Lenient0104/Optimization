
class TransportationNetwork:
    def __init__(self, network, stations, energy_constraints):
        self.network = network  # Location connectivity and time costs
        self.stations = stations  # Stations available at each location
        self.energy_max = energy_constraints  # Max energy available for each mode
        self.pheromone_levels = self.initialize_pheromones()

    def initialize_pheromones(self):
        # Pheromone levels
        pheromone_levels = {}
        # Initialize pheromone levels for all possible paths
        for start_loc in self.network.keys():
            for start_mode in self.stations:
                for end_loc in self.network.keys():
                    for end_mode in self.stations:
                        pheromone_levels[((start_loc, start_mode), (end_loc, end_mode))] = 0.1

        return pheromone_levels

    def calculate_heuristic_info(self):
        # Calculate heuristic information for each path
        heuristic_info = {}
        for key in self.network:
            heuristic_info[key] = []
            for _, _, energy_cost, time_cost in self.network[key][self.stations[0]]:
                heuristic_info[key].append(1 / time_cost)  # or any other heuristic measure
        return heuristic_info

    # Pheromone update function
    def update_pheromones(self, ant, evaporation_rate, pheromone_deposit_amount):
        # Loop through the ant's path to update the pheromones
        for i in range(len(ant.path) - 1):
            current_path = (ant.path[i], ant.path[i + 1])
            if current_path in self.pheromone_levels:
                self.pheromone_levels[current_path] *= (1 - evaporation_rate)
                self.pheromone_levels[current_path] += pheromone_deposit_amount

            # # Update global pheromone levels
            # if current_path in global_pheromone_levels:
            #     global_pheromone_levels[current_path] *= (1 - evaporation_rate)
            #     global_pheromone_levels[current_path] += pheromone_deposit_amount
            # else:
            #     global_pheromone_levels[current_path] = pheromone_deposit_amount

    def pheromone_deposit_function(self, path_cost):
        # Higher deposit for shorter paths
        return 1 / path_cost  # Example function, adjust as needed
