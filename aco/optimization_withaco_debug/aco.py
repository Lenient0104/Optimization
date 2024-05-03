import random


class Ant:

    def __init__(self, start_edge, dest_edge, mode, db_connection, edges):
        self.edges = edges
        self.i = 0
        self.start_edge = start_edge
        self.dest_edge = dest_edge
        self.db_connection = db_connection
        self.vehicle_id, self.energy_level = self.get_best_vehicle_and_energy(mode)  # the vehicle being chosen
        self.initial_energy = {self.vehicle_id: self.energy_level}
        self.remaining_energy = {self.vehicle_id: self.energy_level}
        self.path = [(start_edge, mode, self.vehicle_id)]
        self.total_time_cost = 0
        self.device_to_return = False

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
            if from_edge == '13':
                print("here")
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
        # Abstract move
        current_location, current_mode, current_vehicle_id = self.path[-1]
        edge_found = False
        all_possible_next_moves = self.find_all_next_moves(current_location)
        # print(all_possible_next_moves)
        if len(all_possible_next_moves) == 0:
            self.backtrack()
            return

        # print(all_possible_next_moves)

        # Filter out possible next moves that have the same mode as the current mode
        filtered_possible_next_moves = [move for move in all_possible_next_moves if move[1] == current_mode]
        all_possible_next_edges = [move[0] for move in all_possible_next_moves]
        # Iterate through the edges and check if they exist stations in the table
        # if filtered_possible_next_moves:
        #     all_possible_next_edges = [move[0] for move in filtered_possible_next_moves]
        # Iterate through the edges and check if they exist in the table
        for edge in all_possible_next_edges:
            # Execute a SQL query to check if the edge contains station
            query = "SELECT COUNT(*) FROM StationLocation WHERE StationEdgeID = ?"
            cursor = self.db_connection.cursor()
            cursor.execute(query, (edge,))
            result = cursor.fetchone()
            # Check the result to see if the edge exists
            if result and result[0] > 0:
                edge_found = True
                break  # Exit the loop if any edge is found

        # Check if the there's no available station to change, if so just keep going using the original mode
        # if not edge_found:
        #     # Continue using the current mode and choose the next move based on pheromone level
        #     pheromone_level_max = -1
        #     next_move_max_pheromone = None
        #
        #     for next_move in filtered_possible_next_moves:
        #         next_edge, mode, energy_cost, time_cost = next_move
        #         key = str(str(current_location) + '->' + str(next_move[0]))
        #         pheromone_key_to = (current_mode, 'to')
        #         # Access pheromone level based on the modified key
        #         pheromone_level_to = self.edges[(key)]['pheromone_levels'].get(pheromone_key_to, 0.1)
        #
        #         # choose the maximum pheromone level
        #         if pheromone_level_to > pheromone_level_max:
        #             pheromone_level_max = pheromone_level_to
        #             next_move_max_pheromone = next_move
        #
        #     if next_move_max_pheromone:
        #         self.update_ant_state(next_move_max_pheromone)
        #         print("chosen_move:", next_move)
        #         return
        #     else:
        #         print("No valid path found for moving.")

        # the next possible edges have station options
        # Calculate move probabilities and choose the next move
        move_probabilities = self.calculate_move_probabilities(all_possible_next_moves, 0.2)
        if move_probabilities is None:
            # Apply penalty to the step before last if it exists
            if len(self.path) > 1:
                previous_step = self.path[-2]
                previous_edge = (previous_step[0], current_location)  # Tuple of (from_edge, to_edge)
                self.decrease_pheromones(previous_edge)  # Decrease pheromones for the edge leading to the dead end

        if move_probabilities:
            # Select a move based on probabilities
            chosen_move = random.choices(all_possible_next_moves, weights=list(move_probabilities.values()), k=1)[0]
            print("chosen_move:", chosen_move)
            # next_edge, mode, energy_cost, time_cost = chosen_move
            self.update_ant_state(chosen_move)

            # for move in potential_moves:
            #     if current_mode in ['e-scooter', 'e-bike', 'e-car']:
            #         # Calculate the probability of returning the device
            #         return_device_probability = self.calculate_return_device_probability(move[0])
            #         if random.random() < return_device_probability:
            #             if self.can_return_device(move[0]):
            #                 self.update_ant_state(move)
            #                 return


    def decrease_pheromones(self, edge):
        # Decrease pheromones for the given edge
        evaporation_rate = 0.1  # Adjust as necessary
        penalty_factor = 0.5  # Adjust as necessary, represents how much to decrease
        if edge in self.edges:
            for mode in self.edges[edge]['pheromone_levels']:
                self.edges[edge]['pheromone_levels'][mode]['to'] *= (1 - evaporation_rate) * penalty_factor

    def calculate_move_probabilities(self, next_moves, return_device_probability):
        alpha = 1  # Pheromone importance
        beta = 2  # Heuristic importance
        gamma = 3  # Energy importance
        change_device_rate = 0.2
        edges = self.edges
        probabilities = {}
        current_loc, current_mode, current_vehicle_id = self.path[-1]  # Current location and mode
        # print("current edge:", current_loc)
        # print("current mode:", current_mode)

        for next_move in next_moves:
            next_edge, mode, energy_cost, time_cost = next_move
            has_current_station = self.has_station(next_edge, current_mode)
            if self.is_visited(next_move) or self.is_cycle(next_move):
                probabilities[next_move] = 0
                continue
            if not has_current_station and mode == current_mode and current_mode != 'walking':
                probability = 1
                probabilities[next_move] = probability
                continue
            # there's no current mode's station and we are not walking, so we can only choose to continue use the
            # same mode
            if current_mode != 'walking' and not has_current_station:
                probability = 0
                probabilities[next_move] = probability
                continue

            # initial pheromone level
            # pheromone_key_from = (current_mode, 'from')
            pheromone_key_to = (current_mode, 'to')
            key = str(str(current_loc) + '->' + str(next_move[0]))
            if key in edges:
                # pheromone_level_from = edges[(key)]['pheromone_levels'].get(pheromone_key_from, 0.1)
                pheromone_level_to = edges[(key)]['pheromone_levels'].get(pheromone_key_to, 0.1)
                heuristic = 1 / time_cost if time_cost > 0 else 0  # Avoid division by zero

                # energy factor
                # Special handling for 'walking' mode
                if mode == 'walking':
                    energy_factor = 1.0
                else:
                    energy_factor = self.remaining_energy[current_vehicle_id] / self.initial_energy[
                        current_vehicle_id] if \
                        self.remaining_energy[current_vehicle_id] > 0 else 0

                # Initialize station availability factor
                change_mode = 1

                # Check if the ant will change the device probabilistically
                if random.random() < return_device_probability and current_mode != mode:  # current = scooter, mode = scooter
                    # we consider change the device, while next possible mode is different
                    # If changing device, consider station availability and energy sufficiency
                    if mode == 'e-scooter' and not self.has_scooter_station(next_edge):
                        change_mode = 0
                    elif mode == 'e-bike' and not self.has_bike_station(next_edge):
                        change_mode = 0
                    elif mode == 'e-scooter' and not self.has_bike_station(next_edge):
                        change_mode = 0
                else:
                    # The modes are the same, so no change should occur
                    change_mode = 0

                if change_mode == 1 and current_mode != mode:
                    self.device_to_return = True

                # Calculate the overall probability components
                pheromone_component = pheromone_level_to ** alpha
                heuristic_component = heuristic ** beta
                energy_component = energy_factor ** gamma

                # Calculate the overall probability with the change device rate and station availability factors
                probability = (  # probability for this move
                        pheromone_component * heuristic_component * energy_component * change_device_rate * change_mode
                )

                # If change device rate is low, ensure that the current mode has the highest probability
                if change_device_rate < 1.0 and current_mode == mode:
                    probability = max(probability, 1.0)

                # If the energy is insufficient, set the probability for using the current mode to 0
                if mode == current_mode:
                    if not self.is_energy_enough(next_move):
                        probability = 0
                probabilities[next_move] = probability

        total = sum(probabilities.values())
        if total == 0:
            return
        # if total == 0 and len(probabilities) == 1:
        #     self.update_pheromones(next_moves[0], current_loc)
        #     return
        # Normalize probabilities to sum to 1
        normalized_probabilities = {move: p / total for move, p in probabilities.items()} if total >= 0 else {}
        if len(normalized_probabilities) == 0:
            print("here")
        return normalized_probabilities

    def is_energy_enough(self, next_move):
        current_loc, current_mode, current_vehicle_id = self.path[-1]
        next_edge, mode, energy_cost, time_cost = next_move
        if self.remaining_energy[current_vehicle_id] <= energy_cost:
            return False
        return True

    def is_visited(self, next_move):
        next_edge, mode, energy_cost, time_cost = next_move
        path = self.path

        # Iterate through the path to check if next_edge has been visited
        for edge, _, _ in path:
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

    def has_station(self, edge_id, mode):
        if mode == 'e_bike_1':
            return self.has_bike_station(edge_id)
        if mode == 'e_scooter_1':
            return self.has_scooter_station(edge_id)
        if mode == 'e_car':
            return self.has_car_station(edge_id)
        return False

    def has_scooter_station(self, edge_id):
        station_data = self.get_station_information()
        # Iterate through the station data dictionary
        for station_type, edge_ids in station_data.items():
            for id in edge_ids:
                if edge_id == id and station_type == 'e_scooter_1':
                    return True
        return False

    def has_bike_station(self, edge_id):
        station_data = self.get_station_information()

        # Iterate through the station data dictionary
        for station_type, edge_ids in station_data.items():
            for id in edge_ids:
                if edge_id == id and station_type == 'e_bike_1':
                    return True
        return False

    def has_car_station(self, edge_id):
        station_data = self.get_station_information()

        # Iterate through the station data dictionary
        for station_type, edge_ids in station_data.items():
            for id in edge_ids:
                if edge_id == id and station_type == 'e_car':
                    return True
        return False

    def backtrack(self):
        if len(self.path) > 1:
            self.path.pop()  # Remove the last edge (dead end)
            # Optionally, update pheromones to reflect dead end
            self.update_pheromones_for_dead_end()

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
        current_loc, current_mode, current_vehicle_id = self.path[-1]
        next_edge, mode, energy_cost, time_cost = next_move
        self.update_pheromones(next_move, current_loc)
        # Check if the ant needs to return the device
        if self.device_to_return:
            self.return_device(current_vehicle_id)
            new_vehicle_id, new_energy = self.get_best_vehicle_and_energy(mode)
            self.path.append((next_edge, mode, new_vehicle_id))
            self.remaining_energy[new_vehicle_id] = new_energy
            self.initial_energy[new_vehicle_id] = new_energy
            self.total_time_cost += time_cost

            if mode in ['e_scooter_1', 'e_bike_1', 'e_car']:
                self.device_to_return = False  # reset the flag
            return

        # no change of modes
        self.path.append((next_edge, current_mode, current_vehicle_id))
        self.remaining_energy[current_vehicle_id] -= energy_cost
        self.total_time_cost += time_cost

    # Pheromone update function
    def update_pheromones(self, current_move, pre_edge):
        next_edge, mode, energy_cost, time_cost = current_move
        evaporation_rate = 0.1
        key = str(str(pre_edge) + '->' + str(next_edge))

        if len(self.path) > 2 and next_edge == self.path[-3]:
            # Set pheromone level to 0 for a dead end
            self.edges[(key)]['pheromone_levels'][mode]['from'] = 0
            former_edge, former_mode, _ = self.path[-4]
            new_key = str(str(former_edge) + '->' + str(pre_edge))
            self.edges[(new_key)]['pheromone_levels'][former_mode]['from'] = 0
        else:
            # Update pheromone level based on evaporation and deposit
            self.edges[(key)]['pheromone_levels'][mode]['to'] *= (1 - evaporation_rate)
            self.edges[(key)]['pheromone_levels'][mode]['to'] += self.pheromone_deposit_function(time_cost)

    def pheromone_deposit_function(self, path_cost):
        # Higher deposit for shorter paths
        return 1 / path_cost  # Example function, adjust as needed

