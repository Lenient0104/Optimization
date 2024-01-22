import random
# import sqlite3
#
# import optimization as Optimization
# import user_info as User
#
# net_xml_path = 'graph202311211616.net.xml'
# db_path = 'test_new.db'
# user = User.User(60, False, 0)
# optimizer = Optimization.Optimization(net_xml_path, user, db_path)
# print("Initialize finished")
# edges = optimizer.edges
#
#
# # print(edges)




class Ant:

    def __init__(self, start_edge, dest_edge, mode, db_connection, edges):
        self.edges = edges
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
            cursor.execute(update_query, (vehicle_id, ))

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
                        if int(to_edge) == 13:
                            i = 1
                        # Append the next move as a tuple to the possible_next_moves list
                        if to_edge != current_edge:
                            possible_next_moves.append((to_edge, mode, energy_cost, time_cost))

        return possible_next_moves

    def move(self):
        # Abstract move
        current_location, current_mode, current_vehicle_id = self.path[-1]
        edge_found = False
        all_possible_next_moves = self.find_all_next_moves(current_location)

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
        if not edge_found:
            # Continue using the current mode and choose the next move based on pheromone level
            pheromone_level_max = -1
            next_move_max_pheromone = None

            for next_move in filtered_possible_next_moves:
                next_edge, mode, energy_cost, time_cost = next_move
                key = str(str(current_location) + '->' + str(next_move[0]))
                pheromone_key_to = (current_mode, 'to')
                # Access pheromone level based on the modified key
                pheromone_level_to = self.edges[(key)]['pheromone_levels'].get(pheromone_key_to, 0.1)

                # choose the maximum pheromone level
                if pheromone_level_to > pheromone_level_max:
                    pheromone_level_max = pheromone_level_to
                    next_move_max_pheromone = next_move

            if next_move_max_pheromone:
                self.update_ant_state(next_move_max_pheromone)
            else:
                print("No valid path found for moving.")

        # the next possible edges have station options
        # Calculate move probabilities and choose the next move
        move_probabilities = self.calculate_move_probabilities(all_possible_next_moves, 0.2)
        if move_probabilities:
            # Select a move based on probabilities
            chosen_move = random.choices(all_possible_next_moves, weights=list(move_probabilities.values()), k=1)[0]
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

    def calculate_return_device_probability(self, next_move):
        # Calculate the probability of returning the device
        dest, mode, _, _ = next_move

        # You can customize the probability calculation based on your requirements
        # For example, you can use a fixed probability or a function that depends on the mode or location.
        # Here, we use a simple fixed probability for demonstration purposes.
        return_device_probabilities = {
            'e-scooter': 0.2,  # Example probability for returning an e-scooter
            'e-bike': 0.3,  # Example probability for returning an e-bike
            'e-car': 0.1  # Example probability for returning an e-car
        }

        return return_device_probabilities.get(mode, 0.0)  # Default to 0 if mode not found

    def calculate_move_probabilities(self, next_moves, return_device_probability):
        alpha = 1  # Pheromone importance
        beta = 2  # Heuristic importance
        gamma = 3  # Energy importance
        change_device_rate = 0.2
        edges = self.edges
        probabilities = {}
        current_loc, current_mode, current_vehicle_id = self.path[-1]  # Current location and mode

        for next_move in next_moves:
            next_edge, mode, energy_cost, time_cost = next_move

            # initial pheromone level
            pheromone_key_from = (current_mode, 'from')
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
                    energy_factor = self.remaining_energy[current_vehicle_id] / self.initial_energy[current_vehicle_id] if \
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
        print(f"Total Probability: {total}")  # Debugging print statement

        # Normalize probabilities to sum to 1
        normalized_probabilities = {move: p / total for move, p in probabilities.items()} if total > 0 else {}

        return normalized_probabilities

    def is_energy_enough(self, next_move):
        current_loc, current_mode, current_vehicle_id = self.path[-1]
        next_edge, mode, energy_cost, time_cost = next_move
        if self.remaining_energy[current_vehicle_id] <= energy_cost:
            return False
        return True

    def has_scooter_station(self, edge_id):
        # Check if there is a scooter station available at the 'edge_id'
        if 'station_availability' in self.edges[edge_id]['from'] and self.edges[edge_id]['from']['station_availability'][
            'e_scooter_1']:
            return True
        elif 'station_availability' in self.edges[edge_id]['to'] and self.edges[edge_id]['to']['station_availability'][
            'e_scooter_1']:
            return True
        else:
            return False

    def has_bike_station(self, edge_id):
        # Check if there is a scooter station available at the 'edge_id'
        if 'station_availability' in self.edges[edge_id]['from'] and self.edges[edge_id]['from']['station_availability'][
            'e_bike_1']:
            return True
        elif 'station_availability' in self.edges[edge_id]['to'] and self.edges[edge_id]['to']['station_availability'][
            'e_bike_1']:
            return True
        else:
            return False

    def has_car_station(self, edge_id):
        # Check if there is a scooter station available at the 'edge_id'
        if 'station_availability' in self.edges[edge_id]['from'] and self.edges[edge_id]['from']['station_availability']['e_car']:
            return True
        elif 'station_availability' in self.edges[edge_id]['to'] and self.edges[edge_id]['to']['station_availability']['e_car']:
            return True
        else:
            return False

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
        self.edges[(key)]['pheromone_levels'][mode]['to'] *= (1 - evaporation_rate)
        self.edges[(key)]['pheromone_levels'][mode]['to'] += self.pheromone_deposit_function(time_cost)

    def pheromone_deposit_function(self, path_cost):
        # Higher deposit for shorter paths
        return 1 / path_cost  # Example function, adjust as needed


