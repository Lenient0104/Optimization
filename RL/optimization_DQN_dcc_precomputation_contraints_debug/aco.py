import random


class Ant:

    def __init__(self, start_edge, dest_edge, db_connection, edges, mode_stations, index, graph, mode='walking'):
        self.edges = edges
        self.graph = graph
        self.mode_stations = mode_stations
        self.index = index
        self.path_length = self.find_edge_length(start_edge)
        self.stop = False
        self.start_edge = start_edge
        self.dest_edge = dest_edge
        self.db_connection = db_connection
        self.initial_energy = {}
        self.remaining_energy = {}
        self.reach_dest = False
        if mode == 'walking':
            self.path = [(start_edge, mode, 'pedestrian', 'pedestrian')]
            self.initial_energy['pedestrian'] = 0
            self.remaining_energy['pedestrian'] = 0
        else:
            if self.has_current_station(self.start_edge, mode):
                self.vehicle_id, self.energy_level = self.get_best_vehicle(mode)  # the vehicle being chosen
                self.initial_energy = {str(self.vehicle_id): self.energy_level}
                self.remaining_energy = {str(self.vehicle_id): self.energy_level}
                self.path = [(start_edge, mode, 0, 0)]
            else:
                # print('There are no available station on this edge.')
                self.path = [(start_edge, 'walking', 'pedestrian', 'pedestrian')]
                self.initial_energy['pedestrian'] = 0
                self.remaining_energy['pedestrian'] = 0

        self.total_time_cost = 0
        self.device_to_return = False

    def map_edge_to_length(self):
        edge_distance = {}
        for connection_id, edge_info in self.edges.items():
            # Extract from_edge, to_edge, and length
            from_edge = edge_info['from_edge']
            to_edge = edge_info['to_edge']
            if from_edge not in edge_distance:
                edge_distance[from_edge] = edge_info['from_length']
            if to_edge not in edge_distance:
                edge_distance[to_edge] = edge_info['to_length']
        return edge_distance

    def find_edge_length(self, edge_id):
        edge_distance = self.map_edge_to_length()
        return edge_distance[edge_id]

    def get_best_vehicle_and_energy(self, vehicle_type):
        # Fetch the vehicle with the maximum energy level for the given type
        cursor = self.db_connection.cursor()
        query = """
            SELECT SoCData.VehicleID, SoCData.SoC FROM SoCData
            INNER JOIN VehicleTypes ON SoCData.VehicleID = VehicleTypes.VehicleID
            WHERE VehicleTypes.VehicleType = ?
            ORDER BY SoC DESC, Timestep DESC LIMIT 1
        """
        cursor.execute(query, (vehicle_type,))
        result = cursor.fetchone()
        return (result[0], result[1]) if result else (None, 0)  # vehicle_id, soc

    def check_most_soc(self, vehicle_type):
        most_soc = self.get_best_vehicle_and_energy(vehicle_type)
        return most_soc

    def get_vehicle_energy(self, vehicle_id):
        cursor = self.db_connection.cursor()
        query = """
            SELECT SoC FROM SoCData
            WHERE VehicleID = ?
        """
        cursor.execute(query, (vehicle_id,))
        result = cursor.fetchone()
        return result

    def return_device(self, vehicle_id):
        try:
            cursor = self.db_connection.cursor()

            # Update the SoCData table to set SoC to 100 for the specified edge_id and vehicle_id
            update_query = """
                    UPDATE SoCData
                    SET SoC = 100
                    WHERE VehicleID = ?
                """
            cursor.execute(update_query, (vehicle_id,))

            # Commit the transaction to save the changes
            self.db_connection.commit()

            return True  # Return True to indicate successful update
        except Exception as e:
            # Handle any exceptions (e.g., database errors)
            print(f"Error updating SoC data: {str(e)}")
            return False  # Return False to indicate failure

    def find_all_next_moves(self, current_edge):
        # Initialize an empty list to store possible next moves
        possible_next_moves = []
        graph = self.graph
        for next_edge in graph.neighbors(current_edge):
            edge_data = graph.get_edge_data(current_edge, next_edge)
            for key, edge_data in edge_data.items():
                mode = key
                time = edge_data["weight"]
                distance = edge_data["distance"]
                possible_next_moves.append((next_edge, mode, distance, time))

        return possible_next_moves

    def move(self):
        current_location, current_mode, current_vehicle_id, _ = self.path[-1]
        all_possible_next_moves = self.find_all_next_moves(current_location)
        move_probabilities = self.calculate_move_probabilities(all_possible_next_moves, 0.2, 1)
        # print(move_probabilities)
        if move_probabilities is None:
            self.stop = True
            # Apply penalty to the step before last if it exists
            if len(self.path) > 1:
                previous_step = self.path[-2]
                previous_edge = (previous_step[0], current_location)  # Tuple of (from_edge, to_edge)
                self.decrease_pheromones(previous_edge, current_location,
                                         previous_step[1])  # Decrease pheromones for the edge leading to the dead end
            return

        if move_probabilities:
            chosen_move = random.choices(all_possible_next_moves, weights=list(move_probabilities.values()), k=1)[0]
            chosen_edge, chosen_mode, _, _ = chosen_move
            if chosen_mode != current_mode and current_mode != 'walking':
                self.device_to_return = True
            # print("chosen_move:", chosen_move)
            self.update_ant_state(chosen_move)

    def decrease_pheromones(self, current_edge, previous_edge, mode):
        # Decrease pheromones for the given edge
        evaporation_rate = 0.1  # Adjust as necessary
        penalty_factor = 0.5  # Adjust as necessary, represents how much to decrease
        if self.graph.has_edge(current_edge, previous_edge):
            for edge_data in self.graph.edges[current_edge, previous_edge]:
                if edge_data['mode'] == mode:
                    # Update pheromone level only if the mode matches
                    self.graph.edges[current_edge, previous_edge][edge_data]['pheronmone_level'] = (
                                                                                                               1 - evaporation_rate) * penalty_factor

    # need changing
    def calculate_move_probabilities(self, next_moves, change_device_probability, walking_preference):
        alpha = 1  # Pheromone importance
        beta = 0  # Heuristic importance
        gamma = 1  # Energy importance
        edges = self.edges
        probabilities = {}
        graph = self.graph
        current_loc, current_mode, current_vehicle_id, _ = self.path[-1]  # Current location and mode
        # print("current edge:", current_loc)
        # print("current mode:", current_mode)
        # print("all next moves:", next_moves)
        for next_move in next_moves:
            next_edge, mode, distance, time_cost = next_move

            has_current_station = self.has_current_station(next_edge, current_mode)
            if self.is_visited(next_move) or self.is_cycle(next_move):
                probabilities[next_move] = 0
                continue
            # there's no current mode's station and we are not walking, so we can only choose to continue use the
            # same mode
            if current_mode != 'walking' and not has_current_station and mode == current_mode:
                pheromone_level = graph.get_edge_data(current_loc, next_edge, key=mode)['pheromone_level']
                probability = pheromone_level
                probabilities[next_move] = probability
                continue
            # we are walking and there's no station for change, and we found the next move is walking which is what
            # we want
            if current_mode == 'walking' and not self.has_station(next_edge) and mode == current_mode:
                pheromone_level = graph.get_edge_data(current_loc, next_edge, key=mode)['pheromone_level']
                probability = pheromone_level
                probabilities[next_move] = probability
                continue
            # we are not walking but the next edge doesn't have current station, so keep going
            if current_mode != 'walking' and not has_current_station and next_edge != self.dest_edge:
                probability = 0
                probabilities[next_move] = probability
                continue
            # we are walking and there's no station for change, and we found the next move is walking which is not
            # what we want
            if current_mode == 'walking' and not self.has_station(next_edge):
                # stations = self.stations[next_edge]
                probability = 0
                probabilities[next_move] = probability
                continue
            # --------------------------------- start probability calculation ---------------------------------#
            pheromone_level = graph.get_edge_data(current_loc, next_edge, key=mode)['pheromone_level']

            # heuristic = 1 / time_cost if time_cost > 0 else 0  # Avoid division by zero
            heuristic = 1

            # calculate the possibility to change mode
            if mode != current_mode:
                # energy factor
                # Special handling for changing to 'walking' mode
                if mode == 'walking':
                    energy_factor = 1.0
                    # heuristic = self.find_edge_length(next_edge) / (
                    #         self.path_length + self.find_edge_length(next_edge))
                    heuristic = 1
                else:
                    new_vehicle_id, new_energy_level = self.get_best_vehicle(mode)
                    energy_factor = new_energy_level / 100

                # Calculate the overall probability components
                pheromone_component = pheromone_level ** alpha
                heuristic_component = heuristic ** beta
                energy_component = energy_factor ** gamma
                # Calculate the overall probability with the change device rate and station availability factors
                probability = (  # probability for this move
                        pheromone_component * heuristic_component * energy_component * change_device_probability
                )
                probabilities[next_move] = probability

            # calculate the possibility to not change mode
            elif mode == current_mode:
                if not self.is_energy_enough(next_move):
                    probability = 0
                    probabilities[next_move] = probability
                    continue
                if mode == 'walking':
                    energy_factor = 1.0
                else:
                    energy_factor = self.remaining_energy[current_vehicle_id] / self.initial_energy[
                        current_vehicle_id] if self.remaining_energy[current_vehicle_id] > 0 else 0
                    # energy_factor = 1
                # Calculate the overall probability components
                pheromone_component = pheromone_level ** alpha
                heuristic_component = heuristic ** beta
                energy_component = energy_factor ** gamma
                if mode == 'walking' and current_mode == 'walking':
                    probability = (  # we encourage to use device
                            pheromone_component * heuristic_component * energy_component * change_device_probability
                    )
                    probabilities[next_move] = probability

                else:  # Calculate the overall probability with the change device rate and station availability factors
                    probability = (  # probability for this move
                            pheromone_component * heuristic_component * energy_component * (
                                1 - change_device_probability)
                    )
                    probabilities[next_move] = probability

        total = sum(probabilities.values())
        if total == 0:
            return

        # Normalize probabilities to sum to 1
        normalized_probabilities = {move: p / total for move, p in probabilities.items()} if total >= 0 else {}
        return normalized_probabilities

    def get_best_vehicle(self, mode):
        if mode == 'walking':
            return "pedestrian", "pedestrian"
        if mode == 'e_bike_1':
            e_bike_id = 'eb' + str(random.randint(0, 10))
            soc = random.randint(70, 100)
            return e_bike_id, soc
        elif mode == 'e_scooter_1':
            e_scooter_id = 'es' + str(random.randint(0, 10))
            soc = random.randint(70, 100)
            return e_scooter_id, soc
        elif mode == 'e_car':
            e_car_id = 'ec' + str(random.randint(0, 10))
            soc = random.randint(70, 100)
            return e_car_id, soc

    def calculate_energy_comsumption(self, current_mode, distance):
        # Define vehicle efficiency in Wh per meter (converted from Wh per km)
        vehicle_efficiency = {'e_bike_1': 20 / 1000, 'e_scooter_1': 25 / 1000, 'e_car': 150 / 1000}
        # battery_capacity = {'e_bike_1': 500, 'e_scooter_1': 250, 'e_car': 50000}
        battery_capacity = {'e_bike_1': 500, 'e_scooter_1': 250, 'e_car': 5000}
        energy_consumed = vehicle_efficiency[current_mode] * distance
        # Calculate the delta SoC (%) for the distance traveled
        delta_soc = (energy_consumed / battery_capacity[current_mode]) * 100

        return delta_soc

    def reach_destination(self, current_edge):
        if current_edge == self.dest_edge:
            return True
        return False

    def is_energy_enough(self, next_move):
        current_loc, current_mode, current_vehicle_id, _ = self.path[-1]
        next_edge, mode, distance, time_cost = next_move
        if mode == 'walking':
            return True
        if self.remaining_energy[current_vehicle_id] <= self.calculate_energy_comsumption(current_mode, distance):
            # print("energy not enough")
            return False
        return True

    def is_visited(self, next_move):
        next_edge, mode, distance, time_cost = next_move
        path = self.path

        # Iterate through the path to check if next_edge has been visited
        for edge, _, _, _ in path:
            if edge == next_edge:
                return True  # next_edge has been visited

        return False  # next_edge has not been visited

    def is_cycle(self, next_move):
        # Check if the next move leads to a location that has been recently visited
        if len(self.path) > 2 and (next_move[0] == self.path[-2][0] or next_move[0] == self.path[-3][0]):
            return True
        return False

    def get_station_information(self):
        # cursor = self.db_connection.cursor()
        # query = """
        #             SELECT StationEdgeID, StationType FROM StationLocation
        #         """
        # cursor.execute(query)
        # # Fetch all rows from the query result
        # station_rows = cursor.fetchall()
        # # Initialize an empty dictionary to store the data
        station_data = {}
        # Iterate through the retrieved data and organize it by station type
        for edge_id, station_type in self.stations.items():
            for station in station_type:
                if station not in station_data:
                    station_data[station] = []  # Initialize an empty list for the station type
                station_data[station].append(edge_id)

        return station_data

    def has_current_station(self, edge_id, mode):
        if mode == 'e_bike_1':
            return self.has_bike_station(edge_id)
        if mode == 'e_scooter_1':
            return self.has_scooter_station(edge_id)
        if mode == 'e_car':
            return self.has_car_station(edge_id)
        return False

    def has_station(self, edge_id):
        if self.has_car_station(edge_id) or self.has_bike_station(edge_id) or self.has_scooter_station(edge_id):
            return True
        return False

    def has_scooter_station(self, edge_id):
        station_data = self.mode_stations
        for scooter_edge_id in station_data['e_scooter_1']:
            if edge_id == scooter_edge_id:
                return True
        return False

    def has_bike_station(self, edge_id):
        station_data = self.mode_stations
        for bike_edge_id in station_data['e_bike_1']:
            if edge_id == bike_edge_id:
                return True
        return False

    def has_car_station(self, edge_id):
        station_data = self.mode_stations
        for car_edge_id in station_data['e_car']:
            if edge_id == car_edge_id:
                return True
        return False

    # def update_pheromones(self):
    #     for i in range(0, len(self.path) - 1):
    #         edge_1, mode_1, _, time_1 = self.path[i]
    #         edge_2, mode_2, _, time_2 = self.path[i + 1]
    #         key = str(edge_1) + '->' + str(edge_2)
    #         self.edges[(key)]['pheromone_levels'][mode_1]['from'] += self.pheromone_deposit_function(
    #             self.total_time_cost)
    #         self.edges[(key)]['pheromone_levels'][mode_2]['to'] += self.pheromone_deposit_function(self.total_time_cost)
    #         i = i + 1
    #     return

    def backtrack(self):
        if len(self.path) > 1:
            self.path.pop()  # Remove the last edge (dead end)
            # Optionally, update pheromones to reflect dead end
            self.update_pheromones_for_dead_end()
            self.stop = True

    def update_pheromones_for_dead_end(self):
        if len(self.path) < 2:
            return  # Not enough data to update pheromones

        last_edge = self.path[-1][0]  # Current dead end edge
        pre_edge = self.path[-2][0]  # Edge before the dead end

        # Pheromone update keys
        pre_to_last_mode = self.path[-2][1]
        pheromone_key = (pre_to_last_mode, 'to')

        # Construct the key for the edges dictionary
        edge_key = str(str(pre_edge) + '->' + str(last_edge))

        # Parameters for pheromone update
        evaporation_rate = 0.1  # Rate at which pheromones evaporate
        penalty_factor = 0.5  # Factor to reduce the pheromone level

        if edge_key in self.edges:
            # Decrease the pheromone level with a penalty factor
            self.edges[edge_key]['pheromone_levels'][pre_to_last_mode]['to'] *= (1 - evaporation_rate) * penalty_factor

    def update_ant_state(self, next_move):
        # real move
        current_loc, current_mode, current_vehicle_id, _ = self.path[-1]
        next_edge, mode, distance, time_cost = next_move

        # if the ant needs to return the device or change device
        if self.device_to_return or (current_mode == 'walking' and mode != current_mode):
            if self.device_to_return:
                self.return_device(current_vehicle_id)
            new_vehicle_id, new_energy = self.get_best_vehicle(mode)
            self.path.append((next_edge, mode, new_vehicle_id, new_energy))
            self.remaining_energy[str(new_vehicle_id)] = new_energy
            self.initial_energy[str(new_vehicle_id)] = new_energy
            self.total_time_cost += time_cost
            # self.path_length += self.find_edge_length(next_edge)

            if mode in ['e_scooter_1', 'e_bike_1', 'e_car']:
                self.device_to_return = False  # reset the flag
            return

        # keep walking
        if current_mode == 'walking' and mode == 'walking':
            self.path.append((next_edge, current_mode, current_vehicle_id, 'pedestrian'))
            self.total_time_cost += time_cost
            # self.path_length += self.find_edge_length(next_edge)
            return

        # keep using same device
        self.remaining_energy[current_vehicle_id] -= self.calculate_energy_comsumption(current_mode, distance)
        self.path.append((next_edge, current_mode, current_vehicle_id, self.remaining_energy[current_vehicle_id]))
        # self.path.append((next_edge, current_mode, 0, 0))
        self.total_time_cost += time_cost
        # self.path_length += self.find_edge_length(next_edge)

        return

    def calculate_soc(self, mode, energy):
        if mode == 'e_scooter_1':
            return energy / 446 * 100
        if mode == 'e_bike_1':
            return energy / 450 * 100
        if mode == 'e_car':
            return energy / 75 * 100

    def pheromone_deposit_function(self, path_cost):
        # Higher deposit for faster paths
        return 1 / path_cost  # Example function, adjust as needed
