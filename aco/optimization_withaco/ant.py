# # network:
# # edges data structure
# #             edges[connection_id] = {
# #                 'from_edge': from_edge,
# #                 'to_edge': to_edge,
# #                 'from_length': details_from['length'],
# #                 'to_length': details_to['length'],
# #                 'length': details_from['length'] + details_to['length'],
# #                 'speed_limit': 13.9,
# #                 'mode_time': {
# #                     'e-bike': {'from_time': 0, 'to_time': 0},
# #                     'e-car': {'from_time': 0, 'to_time': 0},
# #                     'e-scooter': {'from_time': 0, 'to_time': 0},
# #                     'walking': {'from_time': 0, 'to_time': 0}
# #                 },
# #                 'speed': {
# #                     'e-bike': {'from_speed': 0, 'to_speed': 0},
# #                     'e-car': {'from_speed': 0, 'to_speed': 0},
# #                     'e-scooter': {'from_speed ': 0, 'to_speed': 0},
# #                     'walking': {'speed': 1.4}
# #                 },
# #                 'from_average_speed': 0,
# #                 'to_average_speed': 0,
# #                 'energy_consumption': {
# #                     'e-bike': {'from_energy': 0, 'to_energy': 0},
# #                     'e-car': {'from_energy': 0, 'to_energy': 0},
# #                     'e-scooter': {'from_energy': 0, 'to_energy': 0}
# #                 },
# #                 'station_availability': {
# #                     'from': {'e-bike': False, 'e-car': False, 'e-scooter': False, 'walking': True},
# #                     'to': {'e-bike': False, 'e-car': False, 'e-scooter': False, 'walking': True}
# #                 },
# #                 'pheromone_levels': {
# #                     'e-bike': {'from': 0.1, 'to': 0.1},
# #                     'e-car': {'from': 0.1, 'to': 0.1},
# #                     'e-scooter': {'from': 0.1, 'to': 0.1},
# #                     'walking': {'from': 0.1, 'to': 0.1}
# #                 }
# #             }
#
# import random
#
#
# # Ant class definition
# class Ant:
#     def __init__(self, start_location, dest_location, vehicle_type, db_connection, network):
#         self.network = network
#         self.start_location = start_location
#         self.dest_location = dest_location
#         self.vehicle_type = vehicle_type
#         self.db_connection = db_connection
#         self.vehicle_id, self.energy_level = self.get_best_vehicle_and_energy() # the vehicle being chosen
#         self.path = [(start_location, self.vehicle_id)]  # Vehicle ID is used instead of type
#         self.total_time_cost = 0
#
#     def get_best_vehicle_and_energy(self):
#         # Fetch the vehicle with the maximum energy level for the given type
#         cursor = self.db_connection.cursor()
#         query = """
#             SELECT VehicleID, SoC FROM SoCData
#             INNER JOIN VehicleTypes ON SoCData.VehicleID = VehicleTypes.VehicleID
#             WHERE VehicleTypes.VehicleType = ?
#             ORDER BY SoC DESC, Timestep DESC LIMIT 1
#         """
#         cursor.execute(query, (self.vehicle_type,))
#         result = cursor.fetchone()
#         return (result[0], result[1]) if result else (None, 0)
#
#     def calculate_move_probabilities(self, next_moves):
#         alpha = 1  # Pheromone importance
#         beta = 2  # Heuristic importance
#         gamma = 3  # Energy importance
#         probabilities = []
#         current_loc, current_mode = self.path[-1]  # Current location and mode
#
#         for next_move in next_moves:
#             dest, mode, energy_cost, time_cost = next_move
#
#             pheromone_key = ((current_loc, current_mode), (dest, mode))
#             pheromone_level = pheromone_levels.get(pheromone_key, 0.1)
#             heuristic = 1 / time_cost if time_cost > 0 else 0  # Avoid division by zero
#             # Special handling for 'walking' mode
#             if mode == 'walking':
#                 energy_factor = 1.0
#             else:
#                 energy_factor = self.remaining_energy[mode] / ENERGY_MAX[mode] if ENERGY_MAX[mode] > 0 else 0
#
#             # Debugging print statements
#             print(
#                 f"Move: {next_move}, Pheromone Level: {pheromone_level}, Heuristic: {heuristic}, Energy Factor: {energy_factor}")
#
#             probability = (pheromone_level ** alpha) * (heuristic ** beta) * (energy_factor ** gamma)
#             probabilities.append(probability)
#
#         total = sum(probabilities)
#         print(f"Total Probability: {total}")  # Debugging print statement
#         return [p / total for p in probabilities] if total > 0 else []
#
#
#     def move(self):
#         current_location, current_mode = self.path[-1]
#
#         # Initialize a list to collect all possible moves
#         all_next_moves = []
#
#         # Add all next moves from the current location
#         for mode in network[current_location]:
#             mode_next_moves = network[current_location][mode]
#             all_next_moves.extend(mode_next_moves)
#
#         # Check if the only available option is 'walking'
#         if len(all_next_moves) == 1 and all_next_moves[0][1] == 'walking':
#             self.update_ant_state(all_next_moves[0])
#             return
#
#         # Calculate move probabilities and choose the next move
#         move_probabilities = self.calculate_move_probabilities(all_next_moves)
#         if move_probabilities:
#             potential_moves = random.choices(all_next_moves, weights=move_probabilities, k=len(all_next_moves))
#
#             # Check if the path eventually leads to a station for returning the device
#             for move in potential_moves:
#                 if current_mode in ['e-scooter', 'e-bike', 'e-car'] and not self.can_return_device(move[0]):
#                     continue  # Skip this move if it doesn't lead to a return station
#                 self.update_ant_state(move)
#                 return
#
#             print("No valid path found for moving or returning the device.")
#
#     def return_device(self):
#         current_location, current_mode = self.path[-1]
#
#         # Check if the current mode is a device that needs to be returned
#         if current_mode in ['e-scooter', 'e-bike', 'e-car']:
#             # Replenish the energy for the returned mode to its initial value
#             self.remaining_energy[current_mode] = ENERGY_MAX[current_mode]
#
#
#
#     def can_return_device(self, location):
#         # Check if there is a path from the given location to a station where the device can be returned
#         if location in network and any(mode in network[location] for mode in ['e-scooter', 'e-bike', 'e-car']):
#             return True
#
#         # Check connections from this location to other stations
#         for mode in network.get(location, {}):
#             for dest, _, _, _ in network[location][mode]:  # Correctly unpack the tuple
#                 if dest in network and any(mode in network[dest] for mode in ['e-scooter', 'e-bike', 'e-car']):
#                     return True
#
#         return False
#
#     def is_cycle(self, next_move):
#         # Check if the next move leads to a location that has been recently visited
#         if len(self.path) > 2 and next_move[0] == self.path[-2][0]:
#             return True
#         return False
#
#
#     def update_ant_state(self, next_station):
#         dest, mode, energy_cost, time_cost = next_station
#
#         # Update the device_to_return flag
#         if mode in ['e-scooter', 'e-bike', 'e-car']:
#             # If the current mode is the same, it means the device is being returned
#             if self.path[-1][1] == mode:
#                 self.device_to_return = False
#             else:
#                 self.device_to_return = True
#         else:
#             self.device_to_return = False
#
#         # Update path, energy, and time cost
#         self.path.append((dest, mode))
#         self.remaining_energy[mode] -= energy_cost
#         self.total_time_cost += time_cost
#
#     def can_move(self):
#         current_location, current_mode = self.path[-1]
#
#         if current_location not in network:
#             print(f"Location {current_location} not found in network")
#             return False
#
#         if not any(self.remaining_energy[mode] > 0 for mode in STATIONS):
#             print("Energy depleted for all modes")
#             return False
#
#         return True
