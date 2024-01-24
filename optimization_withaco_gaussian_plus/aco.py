import random


class Ant:

    def __init__(self, start_edge, dest_edge, db_connection, edges, index, mode='walking'):
        self.edges = edges
        self.index = index
        self.path_length = self.find_edge_length(start_edge)
        self.stop = False
        self.start_edge = start_edge
        self.dest_edge = dest_edge
        self.db_connection = db_connection
        self.initial_energy = {}
        self.remaining_energy = {}
        if mode == 'walking':
            self.path = [(start_edge, mode, 'pedestrian', 'pedestrian')]
            self.initial_energy['pedestrian'] = 0
            self.remaining_energy['pedestrian'] = 0
        else:
            if self.has_current_station(self.start_edge, mode):
                self.vehicle_id, self.energy_level = self.get_best_vehicle_and_energy(mode)  # the vehicle being chosen
                self.initial_energy = {self.vehicle_id: self.energy_level}
                self.remaining_energy = {self.vehicle_id: self.energy_level}
                self.path = [(start_edge, mode, self.vehicle_id, self.energy_level)]
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

    def edge_group(self, all_edges):
        # Initialize an empty dictionary to store the grouped data
        grouped_data = {}

        # Iterate through the data dictionary
        for key, value in all_edges.items():
            from_edge = value['from_edge']
            to_edge = value['to_edge']

            # Check if the 'from_edge' is already a key in the grouped_data dictionary
            if from_edge in grouped_data:
                # Append the 'to_edge' to the existing list
                grouped_data[from_edge].append(to_edge)
            else:
                # Create a new list for this 'from_edge' key
                grouped_data[from_edge] = [to_edge]

        return grouped_data

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
        edges = self.edges
        edge_group = self.edge_group(self.edges)
        edge_index = 0
        # Iterate through the edges dictionary
        for connection_id, edge_data in edges.items():
            if current_edge not in edge_group:
                break
            if edge_index > len(edge_group[current_edge]) - 1:
                break
            from_edge = current_edge
            to_edge = edge_group[from_edge][edge_index]
            edge_index = edge_index + 1
            # Extract relevant data for the next move
            mode_time = edge_data['mode_time']

            # Iterate through transportation modes
            for mode, mode_data in mode_time.items():
                from_time = mode_data['from_time']
                to_time = mode_data['to_time']

                # Ensure that both from_time and to_time are non-negative (valid move)
                if from_time >= 0 and to_time >= 0:
                    # Calculate time_cost
                    time_cost = to_time + from_time

                    # Check if time_cost is non-negative (valid move)
                    if time_cost >= 0:
                        if mode == 'walking':
                            energy_cost = 0
                        else:
                            # Get energy_cost directly from energy_consumption
                            energy_cost = edge_data['energy_consumption'][mode]['from_energy']

                        # Append the next move as a tuple to the possible_next_moves list
                        if to_edge != current_edge:
                            possible_next_moves.append((to_edge, mode, energy_cost, time_cost))

        return possible_next_moves

    def move(self):
        current_location, current_mode, current_vehicle_id, _ = self.path[-1]
        all_possible_next_moves = self.find_all_next_moves(current_location)
        # print(all_possible_next_moves)
        if len(all_possible_next_moves) == 0:
            self.backtrack()
            return
        move_probabilities = self.calculate_move_probabilities(all_possible_next_moves, 0.2, 0.000002)
        if self.index == 900:
            print(move_probabilities)
        if move_probabilities is None:
            # Apply penalty to the step before last if it exists
            if len(self.path) > 1:
                previous_step = self.path[-2]
                previous_edge = (previous_step[0], current_location)  # Tuple of (from_edge, to_edge)
                self.decrease_pheromones(previous_edge)  # Decrease pheromones for the edge leading to the dead end

        if move_probabilities:
            # print(current_location, move_probabilities)
            # Select a move based on probabilities
            chosen_move = random.choices(all_possible_next_moves, weights=list(move_probabilities.values()), k=1)[0]
            chosen_edge, chosen_mode, _, _ = chosen_move
            # if self.index > 500:
            #     if chosen_edge == '-18':
            #         print("hehe")
            if chosen_mode != current_mode and current_mode != 'walking':
                self.device_to_return = True
            # print("chosen_move:", chosen_move)
            self.update_ant_state(chosen_move)
            # edge, _, _, _ = chosen_move
            # if self.reach_destination(edge) and self.total_time_cost < self.best_time:
            #     self.update_pheromones()

    def decrease_pheromones(self, edge):
        # Decrease pheromones for the given edge
        evaporation_rate = 0.1  # Adjust as necessary
        penalty_factor = 0.5  # Adjust as necessary, represents how much to decrease
        if edge in self.edges:
            for mode in self.edges[edge]['pheromone_levels']:
                self.edges[edge]['pheromone_levels'][mode]['to'] *= (1 - evaporation_rate) * penalty_factor

    def calculate_move_probabilities(self, next_moves, change_device_probability, walking_preference):
        alpha = 3  # Pheromone importance
        beta = 0  # Heuristic importance
        gamma = 0  # Energy importance
        edges = self.edges
        probabilities = {}
        current_loc, current_mode, current_vehicle_id, _ = self.path[-1]  # Current location and mode
        # print("current edge:", current_loc)
        # print("current mode:", current_mode)

        for next_move in next_moves:
            next_edge, mode, energy_cost, time_cost = next_move
            has_current_station = self.has_current_station(next_edge, current_mode)
            if self.is_visited(next_move) or self.is_cycle(next_move):
                probabilities[next_move] = 0
                continue
            # there's no current mode's station and we are not walking, so we can only choose to continue use the
            # same mode
            if current_mode != 'walking' and not has_current_station and mode == current_mode:
                key = str(str(current_loc) + '->' + str(next_move[0]))
                pheromone_level_to = edges[(key)]['pheromone_levels'][mode]['to']
                probability = 1 * (1 - walking_preference) * pheromone_level_to
                probabilities[next_move] = probability
                continue
            # we are walking and there's no station for change, and we found the next move is walking which is what
            # we want
            if current_mode == 'walking' and not self.has_station(next_edge) and mode == current_mode:
                key = str(str(current_loc) + '->' + str(next_move[0]))
                pheromone_level_to = edges[(key)]['pheromone_levels'][mode]['to']
                probability = 1 * walking_preference * pheromone_level_to
                probabilities[next_move] = probability
                continue
            # we are not walking but the next edge doesn't have current station, so keep going
            if current_mode != 'walking' and not has_current_station:
                probability = 0
                probabilities[next_move] = probability
                continue
            # we are walking and there's no station for change, and we found the next move is walking which is not
            # what we want
            if current_mode == 'walking' and not self.has_station(next_edge):
                probability = 0
                probabilities[next_move] = probability
                continue

            # initial pheromone level
            # pheromone_key_from = (current_mode, 'from')
            pheromone_key_to = (mode, 'to')
            key = str(str(current_loc) + '->' + str(next_move[0]))
            if key in edges:
                # pheromone_level_from = edges[(key)]['pheromone_levels'].get(pheromone_key_from, 0.1)
                pheromone_level_to = edges[(key)]['pheromone_levels'][mode]['to']
                heuristic = 1 / time_cost if time_cost > 0 else 0  # Avoid division by zero

                # calculate the possibility to change mode
                if mode != current_mode:
                    # energy factor
                    # Special handling for 'walking' mode
                    if mode == 'walking':
                        energy_factor = 1.0
                        heuristic = self.find_edge_length(next_edge) / (
                                    self.path_length + self.find_edge_length(next_edge))
                    else:
                        new_vehicle_id, new_energy_level = self.get_best_vehicle_and_energy(
                            mode)  # the vehicle being chosen
                        energy_factor = new_energy_level / 100

                    # Calculate the overall probability components
                    pheromone_component = pheromone_level_to ** alpha
                    heuristic_component = heuristic ** beta
                    energy_component = energy_factor ** gamma
                    # Calculate the overall probability with the change device rate and station availability factors
                    probability = (  # probability for this move
                            pheromone_component * heuristic_component * energy_component
                    )
                    probabilities[next_move] = probability

                # calculate the possibility to not change mode
                elif mode == current_mode:
                    # If the energy is insufficient, set the probability for using the current mode to 0
                    if not self.is_energy_enough(next_move):
                        probability = 0
                        probabilities[next_move] = probability
                        continue
                    if mode == 'walking':
                        energy_factor = 1.0
                    else:
                        energy_factor = self.remaining_energy[current_vehicle_id] / self.initial_energy[
                            current_vehicle_id] if self.remaining_energy[current_vehicle_id] > 0 else 0
                    # Calculate the overall probability components
                    pheromone_component = pheromone_level_to ** alpha
                    heuristic_component = heuristic ** beta
                    energy_component = energy_factor ** gamma
                    # Calculate the overall probability with the change device rate and station availability factors
                    probability = (  # probability for this move
                            pheromone_component * heuristic_component * energy_component
                    )
                    probabilities[next_move] = probability

        total = sum(probabilities.values())
        if total == 0:
            return

        # Normalize probabilities to sum to 1
        normalized_probabilities = {move: p / total for move, p in probabilities.items()} if total >= 0 else {}
        return normalized_probabilities

    def reach_destination(self, current_edge):
        if current_edge == self.dest_edge:
            return True
        return False

    def is_energy_enough(self, next_move):
        current_loc, current_mode, current_vehicle_id, _ = self.path[-1]
        next_edge, mode, energy_cost, time_cost = next_move
        if mode == 'walking':
            return True
        if self.remaining_energy[current_vehicle_id] <= self.calculate_soc(current_mode, energy_cost):
            return False
        return True

    def is_visited(self, next_move):
        next_edge, mode, energy_cost, time_cost = next_move
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
        cursor = self.db_connection.cursor()
        query = """
                    SELECT StationEdgeID, StationType FROM StationLocation
                """
        cursor.execute(query)
        # Fetch all rows from the query result
        station_rows = cursor.fetchall()
        # Initialize an empty dictionary to store the data
        station_data = {}
        # Iterate through the retrieved data and organize it by station type
        for edge_id, station_type in station_rows:
            if station_type not in station_data:
                station_data[station_type] = []  # Initialize an empty list for the station type
            station_data[station_type].append(edge_id)

        # Close the database connection

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
        station_data = self.get_station_information()
        # Iterate through the station data dictionary
        for station_type, edge_ids in station_data.items():
            for id in edge_ids:
                if int(edge_id) == id and station_type == 'e_scooter_1':
                    return True
        return False

    def has_bike_station(self, edge_id):
        station_data = self.get_station_information()

        # Iterate through the station data dictionary
        for station_type, edge_ids in station_data.items():
            for id in edge_ids:
                if int(edge_id) == id and station_type == 'e_bike_1':
                    return True
        return False

    def has_car_station(self, edge_id):
        station_data = self.get_station_information()

        # Iterate through the station data dictionary
        for station_type, edge_ids in station_data.items():
            for id in edge_ids:
                if int(edge_id) == id and station_type == 'e_car':
                    return True
        return False

    def update_pheromones(self):
        for i in range(0, len(self.path) - 1):
            edge_1, mode_1, _, time_1 = self.path[i]
            edge_2, mode_2, _, time_2 = self.path[i + 1]
            key = str(edge_1) + '->' + str(edge_2)
            self.edges[(key)]['pheromone_levels'][mode_1]['from'] += self.pheromone_deposit_function(
                self.total_time_cost)
            self.edges[(key)]['pheromone_levels'][mode_2]['to'] += self.pheromone_deposit_function(self.total_time_cost)
            i = i + 1
        return

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
        next_edge, mode, energy_cost, time_cost = next_move
        # self.update_pheromones(next_move, current_loc)
        # Check if the ant needs to return the device or change device
        if self.device_to_return or (current_mode == 'walking' and mode != current_mode):
            if self.device_to_return:
                self.return_device(current_vehicle_id)
            new_vehicle_id, new_energy = self.get_best_vehicle_and_energy(mode)
            self.path.append((next_edge, mode, new_vehicle_id, new_energy))
            self.remaining_energy[new_vehicle_id] = new_energy
            self.initial_energy[new_vehicle_id] = new_energy
            self.total_time_cost += time_cost
            self.path_length += self.find_edge_length(next_edge)

            if mode in ['e_scooter_1', 'e_bike_1', 'e_car']:
                self.device_to_return = False  # reset the flag
            return

        # keep using current mode
        if current_mode == 'walking' and mode == 'walking':
            self.path.append((next_edge, current_mode, current_vehicle_id, 'pedestrian'))
            self.total_time_cost += time_cost
            self.path_length += self.find_edge_length(next_edge)
            return

        self.remaining_energy[current_vehicle_id] -= self.calculate_soc(current_mode, energy_cost)
        self.path.append((next_edge, current_mode, current_vehicle_id, self.remaining_energy[current_vehicle_id]))
        self.total_time_cost += time_cost
        self.path_length += self.find_edge_length(next_edge)
        return

    def calculate_soc(self, mode, energy):
        if mode == 'e_scooter_1':
            return energy / 446 * 100
        if mode == 'e_bike_1':
            return energy / 450 * 100
        if mode == 'e_car':
            return energy / 75 * 100

    # # Pheromone update function
    # def update_pheromones(self, current_move, pre_edge):
    #     next_edge, mode, energy_cost, time_cost = current_move
    #     evaporation_rate = 0.1
    #     key = str(str(pre_edge) + '->' + str(next_edge))
    #     if key == '16->48' and mode == 'e_car':
    #         print("hehe")
    #     # Update pheromone level based on evaporation and deposit
    #     # self.edges[(key)]['pheromone_levels'][mode]['to'] *= (1 - evaporation_rate)
    #     self.edges[(key)]['pheromone_levels'][mode]['to'] += self.pheromone_deposit_function(time_cost*10)

    def pheromone_deposit_function(self, path_cost):
        # Higher deposit for faster paths
        return 1 / path_cost  # Example function, adjust as needed
