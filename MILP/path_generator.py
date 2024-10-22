class PathGenerator:
    def __init__(self, G, paths, station_changes, costs, station_change_costs, energy_constraints):
        self.G = G
        self.paths = paths  # Gurobi decision variables for paths
        self.station_changes = station_changes  # Gurobi decision variables for station changes
        self.costs = costs  # Costs associated with paths
        self.station_change_costs = station_change_costs  # Costs associated with station changes
        self.energy_constraints = energy_constraints

    def calculate_fees(self, path):
        fees = 0
        cost_coefficients = {
            'ec': 0.9,  # 电动汽车每公里成本
            'eb': 0.2,  # 电动自行车每公里成本
            'es': 0.1,  # 电动滑板车每公里成本
            'walk': 0  # 步行每公里成本
        }

        # 利润率
        profit_margins = {
            'ec': 1.8,
            'eb': 0.10,
            'es': 0.12,
            'walk': 0
        }

        for sequence in path:
            if len(sequence) == 5:
                i = sequence[0]
                j = sequence[1]
                s = sequence[2]
                dis = self.G[i][j]['weight'] / 1000
                base_cost = cost_coefficients[s] * dis
                fees = fees + base_cost + profit_margins[s] * dis

        return fees

    def calculate_time(self, path):
        time_cost = 0
        for sequence in path:
            t = sequence[3]
            time_cost = time_cost + t
        return time_cost

    def calculate_safety_level(self, path):
        safety = 0
        safety_level = {
            'es': 4,
            'eb': 3,
            'ec': 2,
            'walk': 1
        }
        for sequence in path:
            if len(sequence) == 5:
                i = sequence[0]
                j = sequence[1]
                s = sequence[2]
                dis = self.G[i][j]['weight']
                safety = safety + safety_level[s] * (dis ** 2) * 0.000001
        return safety

    def calculate_walking_distance(self, path):
        walking_distance = 0
        for sequence in path:
            if len(sequence) == 5:
                i = sequence[0]
                j = sequence[1]
                s = sequence[2]
                dis = self.G[i][j]['weight']
                if s == 'walk':
                    walking_distance = walking_distance + dis
                elif s == 'eb' or s == 'es':
                    walking_distance = walking_distance + dis * 0.01
                else:
                    walking_distance = walking_distance + dis * 0.001

        return walking_distance

    def generate_path_sequence(self, start_node, start_station, end_node, end_station):
        current_node, current_mode = start_node, start_station
        path_sequence = []
        energy_consumption_sequence = []  # List to store energy consumption
        station_change_count = 0
        destination_reached = False

        while not destination_reached:
            next_step_found = False

            # Look for the next path step
            for (i, j, s) in self.paths:
                if i == current_node and s == current_mode and self.paths[i, j, s].X == 1:
                    path_cost = self.costs[i, j, s]
                    energy_consumption = self.energy_constraints[i, j, s]
                    path_sequence.append((i, j, s, path_cost, energy_consumption))
                    energy_consumption_sequence.append((i, j, s, energy_consumption))
                    current_node = j
                    next_step_found = True
                    break

            # Look for the next station change
            for (i, s1, s2) in self.station_changes:
                if i == current_node and s1 == current_mode and self.station_changes[i, s1, s2].X == 1:
                    mode_change_cost = self.station_change_costs[i, s1, s2]
                    path_sequence.append((i, s1, s2, mode_change_cost))
                    current_mode = s2
                    station_change_count += 1
                    next_step_found = True

            # Check if destination is reached
            if current_node == end_node and current_mode == end_station:
                destination_reached = True
            elif not next_step_found:
                print("Destination not reached. Path may be incomplete.")
                break

        return path_sequence, station_change_count, self.calculate_fees(path_sequence), \
               self.calculate_time(path_sequence), self.calculate_safety_level(path_sequence), self.calculate_walking_distance(path_sequence)