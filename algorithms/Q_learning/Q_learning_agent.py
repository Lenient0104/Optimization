import csv
import math
import pandas as pd
import random
import time as tm
import matplotlib.pyplot as plt
from datetime import datetime



class MultiModalQLearningAgent:
    def __init__(self, graph, alpha=0.1, gamma=0.98, epsilon=0.1, max_epsilon=1.0, epsilon_decay=0.9999, epsilon_increment=0.01, min_epsilon=0.01):
        self.graph = graph
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.max_epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay  # Exploration decay rate
        self.epsilon_increment = epsilon_increment
        self.min_epsilon = min_epsilon  # Minimum exploration rate
        self.q_table = {}
        self.current_value_column_index = 0
        self.visitedNodes = set()
        self.action_count = {}  # Dictionary to store action selection counts
        self.total_action_count = 0  # Total count of actions taken

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
                    self.action_count[(edge, neighbor, key)] = 0

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

        if random.random() < self.epsilon:
            # Perform a random action
            action = random.choice([action for action in self.q_table if action[0] == state])
        else:
            # Perform the best-known action
            action = max((action for action in self.q_table if action[0] == state), key=lambda x: self.q_table[x],
                         default=None)

            # Increment ε towards the maximum value to increase exploration over time
        self.epsilon = min(self.max_epsilon, self.epsilon + self.epsilon_increment)

        return action

    def get_future_max_q(self, current_state, current_energy, current_mode, depth, energy_rate):
        if depth == 0 or current_state not in self.graph:
            return 0

        future_max_q = float('-inf')
        for neighbor in self.graph[current_state]:
            for mode_key in self.graph[current_state][neighbor]:
                action_key = (current_state, neighbor, mode_key)
                next_distance = self.graph[current_state][neighbor][mode_key]['distance']
                next_energy_consumed = self.calculate_energy_comsumption(mode_key, next_distance)

                # Check for mode change and reset/update energy if necessary
                if mode_key != current_mode:
                    next_energy = 100 * energy_rate
                else:
                    next_energy = current_energy - next_energy_consumed

                if next_energy >= 0:  # Ensure the action is feasible
                    immediate_q = self.q_table.get(action_key, 0)
                    recursive_q = self.get_future_max_q(neighbor, next_energy, mode_key, depth - 1, energy_rate)
                    future_max_q = max(future_max_q, immediate_q + self.gamma * recursive_q)
        return future_max_q

    def update_q_value(self, state, action, reward, current_energy, target_edge, energy_rate):
        _, next_state, mode = action

        distance = self.graph[state][next_state][mode]['weight']  # Assuming 'weight' represents distance
        energy_consumed = self.calculate_energy_comsumption(mode, distance)
        new_energy = current_energy - energy_consumed

        # Use the get_future_max_q function to consider multiple future steps
        future_max_q = self.get_future_max_q(next_state, new_energy, mode, 2, energy_rate)  # Looking ahead 4 steps

        if next_state == target_edge:
            future_max_q = 0

        old_q_value = self.q_table.get(action, 0)

        # Update the Q-value for the current action
        self.q_table[action] = self.q_table.get(action, 0) + \
                               self.alpha * (reward + self.gamma * future_max_q - self.q_table.get(action, 0))

        return old_q_value, self.q_table[action], future_max_q

    def update_epsilon(self):
        # Decay epsilon to reduce exploration over time
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def learn(self, start, destination, episodes, energy_rate, initial_energy, progress_check_interval=100):
        file = open("records_new.csv", "w")
        training_travel_times = []
        for episode in range(1, episodes + 1):
            route = [start]
            self.visitedNodes = set()
            modes = []
            travel_time = 0
            step = 0
            walking_time = 0
            fixed_initial_energy = initial_energy * energy_rate
            # print("initial energy:", fixed_initial_energy)
            current_state = start
            current_energy = fixed_initial_energy  # Initialize the energy level for the vehicle at the start of the episode
            last_mode = None  # Track the last mode used
            done = False
            writer = csv.writer(file)
            writer.writerow(["action", "time",'total time', 'distance', 'energy', 'reward', 'old_q', 'new_q', 'future max', 'done'])
            while not done:
                # print("current state:", current_state)
                action = self.choose_action(current_state, current_energy)

                if action is None:  # No feasible action due to energy constraints
                    print("Ran out of energy, cannot proceed further in this episode.")
                    break

                _, next_state, mode = action

                if next_state in self.visitedNodes:
                    reward = -500
                    old_q_value, new_q, future_max_q = self.update_q_value(current_state, action, reward, current_energy, destination, energy_rate)
                    time = self.graph[current_state][next_state][mode]['weight']
                    travel_time += time
                    distance = self.graph[current_state][next_state][mode]['distance']
                    writer.writerow([action, time, travel_time, distance, current_energy, reward, old_q_value, new_q, future_max_q, done])
                    break
                self.visitedNodes.add(current_state)
                # Check for a mode change, and if so, reset the energy
                if last_mode is not None and mode != last_mode:
                    current_energy = fixed_initial_energy  # Reset energy to maximum on mode change
                last_mode = mode  # Update the last mode used
                # print(mode)
                modes.append(last_mode)

                time = self.graph[current_state][next_state][mode]['weight']

                if mode == 'walking':
                    walking_time += time

                travel_time += time
                distance = self.graph[current_state][next_state][mode]['distance']
                energy_consumed = self.calculate_energy_comsumption(mode, distance)
                # print("energy consumed: ", energy_consumed)
                current_energy -= energy_consumed  # Update energy level after taking the action
                # print("current energy", current_energy)
                # if next_state == destination:
                #     print(route)
                #     print(modes)
                #     print(travel_time)
                if next_state == destination:
                    # if next_state == destination:
                    print(route)
                    print(modes)
                    print(travel_time)
                    reward = 100000 / (travel_time * 0.1)
                    # print("arrived")
                else:
                    reward = 1 / self.graph[current_state][next_state][mode]['weight']

                route.append(current_state)
                step += 1
                pre_state = current_state
                current_state = next_state
                if current_state == destination or current_energy <= 0:
                    done = True
                    old_q_value, new_q, future_max_q = self.update_q_value(pre_state, action, reward, current_energy, destination, energy_rate)
                    writer.writerow([action, time, travel_time, distance, current_energy, reward, old_q_value, new_q, future_max_q, done])
                    training_travel_times.append(travel_time)
                    continue
                old_q_value, new_q, future_max_q = self.update_q_value(pre_state, action, reward, current_energy, destination, energy_rate)
                writer.writerow([action, time, travel_time, distance, current_energy, reward, old_q_value, new_q, future_max_q, done])


            # self.update_epsilon()

            if episode % progress_check_interval == 0:
                print(f"Episode {episode}/{episodes} completed.")
        print("Q-table:", self.q_table)
        file.close()

        plt.plot(training_travel_times)
        plt.show()


    def print_optimal_path(self, start, destination):
        print("Q-table:", self.q_table)
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
    agent.learn(source_edge, target_edge, episode_number, energy_rate, 100)
    best_route, best_modes, time_cost, find = agent.print_optimal_path(source_edge, target_edge)
    time_cost = time_cost + optimizer.edge_map[target_edge]['length'] / 1.5
    end_time = tm.time()
    execution_time = end_time - start_time
    return best_route, best_modes, time_cost, execution_time, find
