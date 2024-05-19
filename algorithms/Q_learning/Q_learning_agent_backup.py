import csv
import random
import time as tm

from matplotlib import pyplot as plt


class MultiModalQLearningAgent:
    def __init__(self, graph, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.01):
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

    def initialize_q_table(self):
        # Initialize Q-values for all state-action pairs
        for edge in self.graph.nodes:
            for neighbor in self.graph.neighbors(edge):
                for key, edge_data in self.graph[edge][neighbor].items():
                    self.q_table[(edge, neighbor, key)] = 0  # Initialize Q-values to 0

    def choose_action(self, state, current_energy):
        actions = [action for action in self.q_table if action[0] == state]
        feasible_actions = []
        for action in actions:
            _, next_state, mode = action
            distance = self.graph[state][next_state][mode]['distance']
            energy_consumed = self.calculate_energy_comsumption(mode, distance)
            # Assuming current_energy is the current energy level of the vehicle
            if current_energy - energy_consumed >= 0:  # Check if the action is feasible
                feasible_actions.append(action)

        if not feasible_actions:  # If no action is feasible due to energy constraints
            return None
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(feasible_actions)
        else:
            state_actions = {action: q for action, q in self.q_table.items() if action in feasible_actions}
            return max(state_actions, key=state_actions.get, default=None)

    def update_q_value(self, state, action, reward, current_energy):
        # Calculate energy consumed for the given action
        _, next_state, mode = action
        distance = self.graph[state][next_state][mode]['distance']  # Assuming 'weight' represents distance
        energy_consumed = self.calculate_energy_comsumption(mode, distance)
        new_energy = current_energy - energy_consumed

        # # Adjust reward based on energy consumption, e.g., penalize high energy consumption
        # energy_penalty = -energy_consumed  # This is a simplistic penalty, consider scaling based on your application
        # adjusted_reward = reward + energy_penalty

        # Initialize next_max Q-value considering feasible actions based on remaining energy
        next_max = float('-inf')
        if next_state in self.graph:
            for neighbor in self.graph[next_state]:
                for mode_key in self.graph[next_state][neighbor]:
                    action_key = (next_state, neighbor, mode_key)
                    if action_key in self.q_table:
                        # Ensure next action is feasible with the remaining energy
                        next_distance = self.graph[next_state][neighbor][mode_key]['distance']
                        next_energy_consumed = self.calculate_energy_comsumption(mode_key, next_distance)
                        if new_energy - next_energy_consumed >= 0:  # Check if the action is feasible with new energy
                            next_max = max(next_max, self.q_table[action_key])

        # If there were no feasible actions from next_state due to energy constraints
        if next_max == float('-inf'):
            next_max = 0

        # Update the Q-value for the current action
        self.q_table[action] = self.q_table.get(action, 0) + \
                               self.alpha * (reward + self.gamma * next_max - self.q_table.get(action, 0))

    def update_epsilon(self):
        # Decay epsilon to reduce exploration over time
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def learn(self, start, destination, episodes, energy_rate, progress_check_interval=100, initial_energy=100):
        total_times = []
        for episode in range(1, episodes + 1):
            step = 0
            total_time = 0
            current_initial_energy = initial_energy * energy_rate
            current_state = start
            current_energy = current_initial_energy  # Initialize the energy level for the vehicle at the start of the episode
            # print('current energy:', current_energy)
            last_mode = None  # Track the last mode used
            done = False
            max_step = 0
            while not done:
                action = self.choose_action(current_state, current_energy)
                if action is None:  # No feasible action due to energy constraints
                    # print("Ran out of energy, cannot proceed further in this episode.")
                    break
                _, next_state, mode = action

                # Check for a mode change, and if so, reset the energy
                if last_mode is not None and mode != last_mode:
                    current_energy = 100  # Reset energy to 100 on mode change
                last_mode = mode  # Update the last mode used

                distance = self.graph[current_state][next_state][mode]['distance']
                time = self.graph[current_state][next_state][mode]['weight']
                total_time += time
                energy_consumed = self.calculate_energy_comsumption(mode, distance)
                current_energy -= energy_consumed  # Update energy level after taking the action
                if step >= 3 and next_state == destination:
                    # print(total_time)
                    total_times.append(total_time)
                    reward = 0


                else:
                    reward = -self.graph[current_state][next_state][mode]['weight']


                pre_state = current_state
                current_state = next_state
                step += 1

                # if current_state == destination and step <= 2:
                #     reward = 10 * (-10000)

                self.update_q_value(pre_state, action, reward, current_energy)
                if current_state == destination or current_energy <= 0:
                    done = True

            self.update_epsilon()

            if episode % progress_check_interval == 0:
                print(f"Episode {episode}/{episodes} completed.")
        #
        # plt.plot(total_times)
        # plt.show()

    def print_optimal_path(self, start, destination):

        current_state = start
        optimal_path = []
        route = [current_state]
        modes = []
        visited_states = set()
        total_time = 0
        find = False

        while current_state != destination:
            if current_state in visited_states:
                print("Detected a loop, cannot find optimal path without revisiting states.")
                break

            visited_states.add(current_state)

            print("current_state", current_state)
            actions = [(action, self.q_table[action]) for action in self.q_table if action[0] == current_state]
            if not actions:
                print("No further actions possible from current state.")
                break

            best_action = max(actions, key=lambda x: x[1])[0]
            _, next_state, mode = best_action
            print()
            modes.append(mode)
            distance = self.graph[current_state][next_state][mode]['distance']
            time = self.graph[current_state][next_state][mode]['weight']
            total_time += time

            optimal_path.append((current_state, mode, next_state))
            current_state = next_state
            route.append(current_state)

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
    best_route, best_modes, time_cost, find = agent.print_optimal_path(source_edge, target_edge)
    time_cost = time_cost + optimizer.edge_map[target_edge]['length'] / 1.5
    end_time = tm.time()
    execution_time = end_time - start_time
    return best_route, best_modes, time_cost, execution_time, find