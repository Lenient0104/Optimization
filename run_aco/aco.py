import random
import time


class Ant:

    def __init__(self, optimizer, start_edge, dest_edge, db_connection, mode_stations, index, graph, energy_rate=1, mode='walking'):
        self.graph = graph
        self.optimizer = optimizer
        self.energy_rate = energy_rate
        self.mode_stations = mode_stations
        self.index = index
        self.stop = False
        self.start_edge = start_edge
        self.dest_edge = dest_edge
        self.db_connection = db_connection
        self.initial_energy = {}
        self.remaining_energy = {}
        self.path = []
        self.current_mode = 'walking'
        self.current_vehicle_id = 'pedestrian'
        self.current_edge = start_edge
        self.reach_dest = False
        # if mode == 'walking':
        #     self.path = [(start_edge, mode, 'pedestrian', 100, 1.5, optimizer.edge_map[start_edge]['length'] / 1.5)]
        self.initial_energy['pedestrian'] = 0
        self.remaining_energy['pedestrian'] = 0
        # else:
        #     if self.has_current_station(self.start_edge, mode):
        #         self.vehicle_id, self.energy_level = self.get_best_vehicle(mode)  # the vehicle being chosen
        #         self.initial_energy = {str(self.vehicle_id): self.energy_level}
        #         self.remaining_energy = {str(self.vehicle_id): self.energy_level}
        #         self.path = [(start_edge, mode, 0, 0)]
        #     else:
        #         # print('There are no available station on this edge.')
        #         self.path = [(start_edge, 'walking', 'pedestrian', 100)]
        #         self.initial_energy['pedestrian'] = 0
        #         self.remaining_energy['pedestrian'] = 0

        self.total_time_cost = 0
        self.device_to_return = False

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
        # current_location, current_mode, current_vehicle_id, _, _, _ = self.path[-1]
        all_possible_next_moves = self.find_all_next_moves(self.current_edge)
        move_probabilities = self.calculate_move_probabilities(all_possible_next_moves, 0.1, 1)
        # print(move_probabilities)
        if move_probabilities is None:
            self.stop = True
            # Apply penalty to the step before last if it exists
            if len(self.path) > 1:
                previous_step = self.path[-2]
                previous_edge = (previous_step[0], self.current_edge)  # Tuple of (from_edge, to_edge)
                self.decrease_pheromones(previous_edge, self.current_edge,
                                         previous_step[1])  # Decrease pheromones for the edge leading to the dead end
            return

        if move_probabilities:
            chosen_move = random.choices(all_possible_next_moves, weights=list(move_probabilities.values()), k=1)[0]
            chosen_edge, chosen_mode, _, _ = chosen_move
            if chosen_mode != self.current_mode and self.current_mode != 'walking':
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
        probabilities = {}
        graph = self.graph
        # current_loc, current_mode, current_vehicle_id, _ = self.path[-1]  # Current location and mode
        # print("current edge:", current_loc)
        # print("current mode:", current_mode)
        # print("all next moves:", next_moves)
        for next_move in next_moves:
            next_edge, mode, distance, time_cost = next_move

            has_current_station = self.has_current_station(next_edge, self.current_mode)
            if self.is_visited(next_move) or self.is_cycle(next_move):
                probabilities[next_move] = 0
                continue
            # there's no current mode's station and we are not walking, so we can only choose to continue use the
            # same mode
            if self.current_mode != 'walking' and not has_current_station and mode == self.current_mode:
                pheromone_level = graph.get_edge_data(self.current_edge, next_edge, key=mode)['pheromone_level']
                probability = pheromone_level * 0
                probabilities[next_move] = probability
                continue
            # we are walking and there's no station for change, and we found the next move is walking which is what
            # we want
            if self.current_mode == 'walking' and not self.has_station(next_edge) and mode == self.current_mode:
                pheromone_level = graph.get_edge_data(self.current_edge, next_edge, key=mode)['pheromone_level']
                probability = pheromone_level * 0
                probabilities[next_move] = probability
                continue
            # we are not walking but the next edge doesn't have current station, so keep going
            if self.current_mode != 'walking' and not has_current_station and next_edge != self.dest_edge:
                probability = 0
                probabilities[next_move] = probability
                continue
            # we are walking and there's no station for change, and we found the next move is walking which is not
            # what we want
            if self.current_mode == 'walking' and not self.has_station(next_edge):
                # stations = self.stations[next_edge]
                probability = 0
                probabilities[next_move] = probability
                continue

            # --------------------------------- start probability calculation ---------------------------------#
            pheromone_level = graph.get_edge_data(self.current_edge, next_edge, key=mode)['pheromone_level']

            # heuristic = 1 / time_cost if time_cost > 0 else 0  # Avoid division by zero
            heuristic = 1

            # calculate the possibility to change mode
            if mode != self.current_mode:
                # energy factor
                # Special handling for changing to 'walking' mode
                if mode == 'walking':
                    energy_factor = 1.0
                    # heuristic = self.find_edge_length(next_edge) / (
                    #         self.path_length + self.find_edge_length(next_edge))
                    heuristic = 1
                else:
                    _, new_energy_level = self.get_best_vehicle(mode)
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
            elif mode == self.current_mode:
                if not self.is_energy_enough(next_move):
                    probability = 0
                    probabilities[next_move] = probability
                    continue
                if mode == 'walking':
                    energy_factor = 1.0
                else:
                    energy_factor = self.remaining_energy[self.current_vehicle_id] / self.initial_energy[
                        self.current_vehicle_id] if self.remaining_energy[self.current_vehicle_id] > 0 else 0
                    # energy_factor = 1
                # Calculate the overall probability components
                pheromone_component = pheromone_level ** alpha
                heuristic_component = heuristic ** beta
                energy_component = energy_factor ** gamma
                if mode == 'walking' and self.current_mode == 'walking':
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
            return "pedestrian", 100
        if mode == 'e_bike_1':
            e_bike_id = 'eb' + str(random.randint(0, 10))
            soc = 100 * self.energy_rate
            return e_bike_id, soc
        elif mode == 'e_scooter_1':
            e_scooter_id = 'es' + str(random.randint(0, 10))
            soc = 100 * self.energy_rate
            return e_scooter_id, soc
        elif mode == 'e_car':
            e_car_id = 'ec' + str(random.randint(0, 10))
            soc = 100 * self.energy_rate
            return e_car_id, soc

    def calculate_energy_comsumption(self, current_mode, distance):
        if current_mode == 'walking':
            return 0
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
        # current_loc, current_mode, current_vehicle_id, _ = self.path[-1]
        next_edge, mode, distance, time_cost = next_move
        if mode == 'walking':
            return True
        if self.remaining_energy[self.current_vehicle_id] <= self.calculate_energy_comsumption(self.current_mode, distance):
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

    # def get_station_information(self):
    #     # cursor = self.db_connection.cursor()
    #     # query = """
    #     #             SELECT StationEdgeID, StationType FROM StationLocation
    #     #         """
    #     # cursor.execute(query)
    #     # # Fetch all rows from the query result
    #     # station_rows = cursor.fetchall()
    #     # # Initialize an empty dictionary to store the data
    #     station_data = {}
    #     # Iterate through the retrieved data and organize it by station type
    #     for edge_id, station_type in self.stations.items():
    #         for station in station_type:
    #             if station not in station_data:
    #                 station_data[station] = []  # Initialize an empty list for the station type
    #             station_data[station].append(edge_id)
    #
    #     return station_data

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

    def update_ant_state(self, next_move):
        # real move
        # print('remaning energy', self.remaining_energy)
        # current_loc, current_mode, current_vehicle_id, _ = self.path[-1]
        next_edge, mode, distance, time_cost = next_move


        # if the ant needs to return the device or change device
        if self.device_to_return or (self.current_mode == 'walking' and mode != self.current_mode):
            # if self.device_to_return:
            #     self.return_device(current_vehicle_id)
            new_vehicle_id, new_energy = self.get_best_vehicle(mode)
            self.path.append((self.current_edge, mode, new_vehicle_id, new_energy))
            self.current_edge = next_edge
            self.current_mode = mode
            self.current_vehicle_id = new_vehicle_id
            self.remaining_energy[str(new_vehicle_id)] = new_energy
            self.initial_energy[str(new_vehicle_id)] = new_energy
            self.total_time_cost += time_cost
            # self.path_length += self.find_edge_length(next_edge)

            if mode in ['e_scooter_1', 'e_bike_1', 'e_car']:
                self.device_to_return = False  # reset the flag
            return

        # keep walking
        if self.current_mode == 'walking' and mode == 'walking':
            self.path.append((self.current_edge, self.current_mode, self.current_vehicle_id, 100))
            self.current_edge = next_edge
            self.total_time_cost += time_cost
            # self.path_length += self.find_edge_length(next_edge)
            return

        # keep using same device
        self.remaining_energy[self.current_vehicle_id] -= self.calculate_energy_comsumption(self.current_mode, distance)
        self.path.append((self.current_edge, self.current_mode, self.current_vehicle_id, self.remaining_energy[self.current_vehicle_id]))
        self.current_edge = next_edge
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


def update_pheromones(ant, graph):
    """

    :param graph:
    :param ant:
    :return:
    """
    total_pheromone = 0
    for i in range(0, len(ant.path) - 1):
        edge_1, mode_1, _, _ = ant.path[i]
        edge_2, mode_2, _, _ = ant.path[i + 1]
        # Retrieve all edge data between edge_1 and edge_2
        all_edges_data = graph.get_edge_data(edge_1, edge_2)
        key = None
        # Check if there are any edges between these nodes
        if all_edges_data:
            for key, edge_data in all_edges_data.items():
                if key == mode_2:
                    # Correctly access and update pheromone level
                    current_pheromone_level = edge_data.get('pheromone_level',
                                                            0)  # Get current level, default to 0 if not set
                    # Update pheromone level
                    updated_pheromone_level = current_pheromone_level + pheromone_deposit_function(
                        ant.total_time_cost)
                    # Set the updated pheromone level back on the edge
                    graph[edge_1][edge_2][key]['pheromone_level'] = updated_pheromone_level

        total_pheromone = total_pheromone + graph[edge_1][edge_2][key]['pheromone_level']
    return total_pheromone


def pheromone_deposit_function(path_cost):
    # Higher deposit for faster paths
    return 1 / path_cost  # Example function, adjust as needed


def find_best_path(ants, destination_edge, pheromones):
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


def run_aco_algorithm(optimizer, start_edge, destination_edge, number_of_ants, energy_rate, mode='walking'):
    start_time = time.time()
    if start_edge not in optimizer.unique_edges:
        print(f"There's no edge {start_edge} in this map.")
        return
    if destination_edge not in optimizer.unique_edges:
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
        ant = Ant(optimizer, start_edge, destination_edge, optimizer.db_connection, optimizer.mode_stations, ant_index,
                  optimizer.new_graph, energy_rate, mode)
        print("ant", ant_index)
        move_count = 0
        while ant.current_edge != destination_edge and ant.stop is False:
            # and ant.total_time_cost <= best_time_cost
            ant.move()
            # print(ant.path)
            if len(ant.path) > 2 and (ant.path[-1][0] == ant.path[-2][0]):
                break
            move_count += 1
        # if ant_num == 2:
        #     print('hehe')
            if move_count > 1000:
                # print("Move limit reached, breaking out of loop.")
                break
        # time_costs.append(ant.total_time_cost)
        # print(ant.path)
        if ant.current_edge == destination_edge:
            ant.path.append((ant.current_edge, ant.current_mode, ant.current_vehicle_id, ant.remaining_energy['pedestrian']))
            # print(ant.path)
            print("time cost:", ant.total_time_cost)

            if ant.total_time_cost <= best_time_cost:
                # print("curren best time cost", best_time_cost)
                # print("curren time cost", ant.total_time_cost)
                time_costs.append(ant.total_time_cost)
                # print(ant.total_time_cost)
                ants.append(ant)
                best_time_cost = ant.total_time_cost
                total_pheromone = optimizer.update_pheromones(ant)
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
    optimizer.reset_graph()
    current_best_ant = optimizer.find_best_path(ants, destination_edge, total_pheromones)  # based on pheromone level
    if current_best_ant is not None:
        best_path = current_best_ant.path
        best_time_cost = current_best_ant.total_time_cost + optimizer.edge_map[destination_edge]['length'] / 1.5
    # best_distance_cost = current_best_ant.path_length

    # if (iteration + 1) % log_interval == 0 or iteration == number_of_iterations - 1: print( f"Iteration {
    # iteration + 1}/{number_of_iterations}: Best Path = {best_path}, Time Cost = {best_time_cost},
    # Total Distance = {best_distance_cost}")

    if not flag:
        end_time = time.time()
        exe_time = end_time - start_time
    # self.visualization(time_costs, number_of_ants)
    expanded_path = []
    for i in range(0, len(best_path) - 1):
        current_edge = best_path[i][0]
        next_edge = best_path[i+1][0]
        mode = best_path[i][1]
        vehicle_id = best_path[i][2]
        begin_energy = best_path[i][3]
        path = optimizer.new_graph[current_edge][next_edge][mode]['path']
        for edge in path:
            mode_key = None
            for edge in path:
                unit_energy_consumption = ant.calculate_energy_comsumption(mode, optimizer.edge_map[edge]['length'])
                if mode == 'walking':
                    mode_key = "pedestrian_speed"
                elif mode == 'e_bike_1' or mode == 'e_scooter_1':
                    mode_key = "bike_speed"
                elif mode == 'e_car':
                    mode_key = "car_speed"
                edge_speed = float(optimizer.simulation_data[edge][mode_key])
                if edge_speed == 0:
                    unit_time = 50
                else:
                    unit_time = optimizer.edge_map[edge]['length'] / edge_speed
                unit_remaining_energy = begin_energy - unit_energy_consumption
                begin_energy = unit_remaining_energy
                expanded_path.append((edge, mode, vehicle_id, unit_remaining_energy, edge_speed, unit_time))

    return expanded_path, best_time_cost, exe_time
