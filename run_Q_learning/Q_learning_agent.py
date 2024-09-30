import csv
import random
import time as tm

from matplotlib import pyplot as plt


class MultiModalQLearningAgent:
    def __init__(self, graph, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.01):
        self.graph = graph
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Exploration decay rate
        self.min_epsilon = min_epsilon  # Minimum exploration rate
        self.q_table = {}

        self.initialize_q_table()

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

    def get_best_vehicle(self, mode):
        if mode == 'walking':
            return "pedestrian"
        if mode == 'e_bike_1':
            e_bike_id = 'eb' + str(random.randint(0, 10))
            soc = random.randint(90, 100)
            return e_bike_id
        elif mode == 'e_scooter_1':
            e_scooter_id = 'es' + str(random.randint(0, 10))
            soc = random.randint(90, 100)
            return e_scooter_id
        elif mode == 'e_car':
            e_car_id = 'ec' + str(random.randint(0, 10))
            soc = random.randint(90, 100)
            return e_car_id


    def initialize_q_table(self):
        # Initialize Q-values for all state-action pairs
        for edge in self.graph.nodes:
            for neighbor in self.graph.neighbors(edge):
                for key, edge_data in self.graph[edge][neighbor].items():
                    self.q_table[(edge, neighbor, key)] = -200  # Initialize Q-values to 0

    def choose_action(self, state, current_energy):
        chosen_action = None
        actions = [action for action in self.q_table if action[0] == state]
        # feasible_actions = []
        # for action in actions:
        #     _, next_state, mode = action
        #     distance = self.graph[state][next_state][mode]['distance']
        #     energy_consumed = self.calculate_energy_comsumption(mode, distance)
        #     # Assuming current_energy is the current energy level of the vehicle
        #     if current_energy - energy_consumed >= 0:  # Check if the action is feasible
        #         feasible_actions.append(action)
        #
        # if not feasible_actions:  # If no action is feasible due to energy constraints
        #     return None
        if random.uniform(0, 1) < self.epsilon:
            # print("random")
            chosen_action = random.choice(actions)
        else:
            state_actions = {action: q for action, q in self.q_table.items() if action in actions}
            chosen_action = max(state_actions, key=state_actions.get, default=None)
            # print("max")
        # print('chosen action: ', chosen_action)
        return chosen_action

    def update_q_value(self, state, action, reward, current_energy, energy_rate):
        _, next_state, current_mode = action
        distance = self.graph[state][next_state][current_mode]['distance']
        energy_consumed = self.calculate_energy_comsumption(current_mode, distance)
        new_energy = current_energy - energy_consumed

        next_max = float('-inf')
        if next_state in self.graph:
            for neighbor in self.graph[next_state]:
                for mode_key in self.graph[next_state][neighbor]:
                    action_key = (next_state, neighbor, mode_key)
                    if action_key in self.q_table:
                        next_distance = self.graph[next_state][neighbor][mode_key]['distance']
                        next_energy_consumed = self.calculate_energy_comsumption(mode_key, next_distance)

                        if mode_key != current_mode:
                            new_energy = 100 * energy_rate

                        next_energy = new_energy - next_energy_consumed

                        if next_energy >= 0:
                            next_max = max(next_max, self.q_table[action_key])

        if next_max == float('-inf'):
            next_max = 0

        old_q = self.q_table.get(action, 0)

        self.q_table[action] = self.q_table.get(action, 0) + \
                               self.alpha * (reward + self.gamma * next_max - self.q_table[action])

        new_q = self.q_table[action]


        return old_q, new_q



    def update_epsilon(self):
        # Decay epsilon to reduce exploration over time
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def learn(self, start, destination, episodes, energy_rate, progress_check_interval=100, initial_energy=100):
        total_times = []
        # file = open("records_new.csv", "w")
        for episode in range(1, episodes + 1):
            step = 0
            route = []
            modes = []
            total_time = 0
            current_initial_energy = initial_energy * energy_rate
            # print("current energy is", current_initial_energy)
            current_state = start
            route.append(current_state)
            current_energy = current_initial_energy  # Initialize the energy level for the vehicle at the start of the episode
            # print('current energy:', current_energy)
            last_mode = None  # Track the last mode used
            done = False
            # writer = csv.writer(file)
            # writer.writerow(
            #     ["action", "time", 'total time', 'distance', 'energy', 'reward', 'old_q', 'new_q',
            #      'done', 'step'])
            max_step = 0
            while not done:
                action = self.choose_action(current_state, current_energy)
                if action is None:  # No feasible action due to energy constraints
                    print("Ran out of energy, cannot proceed further in this episode.")
                    break
                _, next_state, mode = action

                modes.append(mode)

                # Check for a mode change, and if so, reset the energy
                if last_mode is not None and mode != last_mode:
                    current_energy = current_initial_energy  # Reset energy to 100 on mode change
                last_mode = mode  # Update the last mode used

                distance = self.graph[current_state][next_state][mode]['distance']
                time = self.graph[current_state][next_state][mode]['weight']
                total_time += time
                energy_consumed = self.calculate_energy_comsumption(mode, distance)
                # print("energy consumed: ", energy_consumed)
                current_energy -= energy_consumed  # Update energy level after taking the action
                # print(current_energy)
                if step >= 3 and next_state == destination:
                    # total_times.append(total_time)
                    reward = 1 / total_time


                else:
                    reward = -self.graph[current_state][next_state][mode]['weight']

                if current_energy < 0:
                    reward = -10000000


                pre_state = current_state
                current_state = next_state
                route.append(current_state)


                if current_state == destination:
                    # print(route)
                    # print(modes)
                    # print(total_time)
                    done = True

                # if current_state == destination and step <= 2:
                #     reward = 10 * (-10000)

                old_q, new_q = self.update_q_value(pre_state, action, reward, current_energy, energy_rate)
                # writer.writerow(
                #     [action, time, total_time, distance, current_energy, reward, old_q, new_q,
                #      done, step])
                step += 1

            # print(self.epsilon)
            self.update_epsilon()

            if episode % progress_check_interval == 0:
                print(f"Episode {episode}/{episodes} completed.")
        #
        # plt.plot(total_times)
        # plt.show()
        # file.close()

    def print_optimal_path(self, optimizer, start, destination):

        current_state = start
        optimal_path = []
        edges = []
        route = [current_state]
        modes = []
        last_mode = None
        vehicle_id = None
        visited_states = set()
        total_time = 0
        find = False
        initial_energy = 100

        while current_state != destination:
            if current_state in visited_states:
                print("Detected a loop, cannot find optimal path without revisiting states.")
                break

            visited_states.add(current_state)
            #
            # print("current_state", current_state)
            actions = [(action, self.q_table[action]) for action in self.q_table if action[0] == current_state]
            if not actions:
                print("No further actions possible from current state.")
                break

            best_action = max(actions, key=lambda x: x[1])[0]
            _, next_state, mode = best_action
            modes.append(mode)
            distance = self.graph[current_state][next_state][mode]['distance']
            time = self.graph[current_state][next_state][mode]['weight']
            path = self.graph[current_state][next_state][mode]['path']
            if last_mode != mode:
                initial_energy = 100
                vehicle_id = self.get_best_vehicle(mode)
            energy_consumed = self.calculate_energy_comsumption(mode, distance)
            begin_energy = initial_energy
            initial_energy = initial_energy - energy_consumed
            total_time += time
            mode_key = None
            for edge in path[:-1]:
                unit_energy_consumption = self.calculate_energy_comsumption(mode, optimizer.edge_map[edge]['length'])
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
                unit_remaining_energy = begin_energy-unit_energy_consumption
                begin_energy = unit_remaining_energy
                optimal_path.append((edge, mode, vehicle_id, unit_remaining_energy, edge_speed, unit_time))
                edges.append(edge)
            current_state = next_state
            route.append(current_state)
            last_mode = mode
        print('optimal path is:', optimal_path)
        print('edges:', edges)

        # Join array elements with a space
        formatted_edges = " ".join(edges)

        # Define the output file name
        output_file = "formatted_edges.txt"

        # Write the formatted edges to the output file
        with open(output_file, "w") as file:
            file.write(formatted_edges)

        print(f"Formatted edges have been written to '{output_file}'")

        if current_state == destination:
            # print("Best route:")
            # for step in optimal_path:
            #     print(f"{step[0]} --[{step[1]}]--> {step[2]}")
            # print(f"total time: {total_time} seconds")
            find = True

        else:
            print("Failed to start the path; check the starting state and Q-table.")
        return route, modes, total_time, find


def run_q_learning(optimizer, source_edge, target_edge, episode_number, energy_rate):
    agent = MultiModalQLearningAgent(optimizer.new_graph)
    start_time = tm.time()
    agent.learn(source_edge, target_edge, episode_number, energy_rate)
    best_route, best_modes, time_cost, find = agent.print_optimal_path(optimizer, source_edge, target_edge)
    time_cost = time_cost + optimizer.edge_map[target_edge]['length'] / 1.5
    end_time = tm.time()
    execution_time = end_time - start_time
    return best_route, best_modes, time_cost, execution_time, find