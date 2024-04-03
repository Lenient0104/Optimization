import random
import statistics
import time as tm

from matplotlib import pyplot as plt

from user_info import User
from optimization import Optimization


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
            distance = self.graph[state][next_state][mode]['weight']
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
        distance = self.graph[state][next_state][mode]['weight']  # Assuming 'weight' represents distance
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
                        next_distance = self.graph[next_state][neighbor][mode_key]['weight']
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

    def learn(self, start, destination, episodes, progress_check_interval=100, initial_energy=100):
        for episode in range(1, episodes + 1):
            current_state = start
            current_energy = initial_energy  # Initialize the energy level for the vehicle at the start of the episode
            last_mode = None  # Track the last mode used
            done = False
            while not done:
                action = self.choose_action(current_state, current_energy)
                if action is None:  # No feasible action due to energy constraints
                    print("Ran out of energy, cannot proceed further in this episode.")
                    break
                _, next_state, mode = action

                # Check for a mode change, and if so, reset the energy
                if last_mode is not None and mode != last_mode:
                    current_energy = 100  # Reset energy to 100 on mode change
                last_mode = mode  # Update the last mode used

                distance = self.graph[current_state][next_state][mode]['weight']
                energy_consumed = self.calculate_energy_comsumption(mode, distance)
                current_energy -= energy_consumed  # Update energy level after taking the action

                reward = -self.graph[current_state][next_state][mode]['weight']

                self.update_q_value(current_state, action, reward, current_energy)

                current_state = next_state
                if current_state == destination or current_energy <= 0:
                    done = True

            self.update_epsilon()

            if episode % progress_check_interval == 0:
                print(f"Episode {episode}/{episodes} completed.")

    def print_optimal_route(self, start, destination):
        current_state = start
        visited_states = {start}  # Initialize with the start state
        optimal_route = [start]
        modes_used = []
        time_cost = 0
        total_energy_consumed = 0  # Initialize total energy consumed

        print("Optimal Route, Modes, and Energy Consumption:")

        while current_state != destination:
            actions = [(action, self.q_table[action]) for action in self.q_table if
                       action[0] == current_state and action[1] not in visited_states]

            if not actions:  # Check if there are no viable actions
                print("No further actions available. Route may be incomplete.")
                break

            best_action = max(actions, key=lambda x: x[1])[0]
            _, next_state, mode = best_action

            if next_state in visited_states:  # Additional check for cycles
                print("Detected a cycle with visited states. Breaking out.")
                break

            # Retrieve the distance for the current segment of the route
            distance = self.graph.get_edge_data(current_state, next_state, key=mode)['weight']
            time_cost += distance

            # Calculate energy consumption for the current segment
            energy_consumed = self.calculate_energy_comsumption(mode, distance)
            total_energy_consumed += energy_consumed  # Accumulate total energy consumed

            optimal_route.append(next_state)
            modes_used.append(mode)
            visited_states.add(next_state)  # Mark the next_state as visited
            current_state = next_state

        # Print the optimal route, modes used, and energy consumption for each segment
        for i in range(len(optimal_route) - 1):
            distance = self.graph.get_edge_data(optimal_route[i], optimal_route[i + 1], key=modes_used[i])['weight']
            energy_consumed = self.calculate_energy_comsumption(modes_used[i], distance)
            print(
                f"{optimal_route[i]} -> {optimal_route[i + 1]} (Mode: {modes_used[i]}, Distance: {distance}m, Energy Consumed: {energy_consumed:.2f}%)")
        print(
            f"Destination reached: {destination}\nTotal Time: {time_cost} s\nTotal Energy Consumed: {total_energy_consumed:.2f}%")
        return time_cost, total_energy_consumed


if __name__ == '__main__':
    net_xml_path = 'DCC.net.xml'
    source_edge = '361450282'
    target_edge = "-110407380#1"
    user = User(60, False, 0, 20)
    episodes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
    test_size = 30
    Q_exe_times = []
    Q_times = []

    for episode in episodes:
        episode_exe_times = []
        episode_times = []
        for _ in range(test_size):
            optimizer = Optimization(net_xml_path, user, 'test_new.db', source_edge, target_edge)
            graph = optimizer.new_graph
            agent = MultiModalQLearningAgent(graph)
            start_time = tm.time()
            agent.learn(source_edge, target_edge, episode)
            end_time = tm.time()
            _, time_cost = agent.print_optimal_route(source_edge, target_edge)
            episode_exe_times.append(end_time - start_time)
            episode_times.append(time_cost)
        Q_exe_times.append(episode_exe_times)
        Q_times.append(episode_times)

    plt.rcParams.update({'font.size': 14})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    for idx, episode in enumerate(episodes):
        avg_exe_time = statistics.mean(Q_exe_times[idx])
        exe_time_diffs = [t - avg_exe_time for t in Q_exe_times[idx]]
        ax1.plot([episode] * len(exe_time_diffs), exe_time_diffs, 'o-')

        avg_time = statistics.mean(Q_times[idx])
        time_diffs = [t - avg_time for t in Q_times[idx]]
        ax2.plot([episode] * len(time_diffs), time_diffs, 'o-')

    ax1.set_xlabel('Number of Episodes', fontsize=16)
    ax1.set_ylabel('Execution Time Difference (seconds)', fontsize=16)
    ax1.set_title('Execution Time Variability per Episode', fontsize=18)
    # ax1.tick_params(axis='both', which='major', labelsize=14)
    # ax1.legend()
    ax1.grid(True)

    ax2.set_xlabel('Number of Episodes', fontsize=16)
    ax2.set_ylabel('Time Cost Difference (seconds)', fontsize=16)
    ax2.set_title('Time Cost Variability per Episode', fontsize=18)
    # ax2.tick_params(axis='both', which='major', labelsize=14)
    # ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

