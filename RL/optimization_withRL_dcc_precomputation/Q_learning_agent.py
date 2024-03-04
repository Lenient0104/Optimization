import networkx as nx
import numpy as np
import random
import time
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

    def initialize_q_table(self):
        # Initialize Q-values for all state-action pairs
        for edge in self.graph.nodes:
            for neighbor in self.graph.neighbors(edge):
                for key, edge_data in self.graph[edge][neighbor].items():
                    self.q_table[(edge, neighbor, key)] = 0  # Initialize Q-values to 0

    def choose_action(self, state):
        actions = [action for action in self.q_table if action[0] == state]
        if not actions:
            return None
        # If there are actions, proceed with epsilon-greedy policy
        if random.uniform(0, 1) < self.epsilon:
            # Exploration: choose a random action
            return random.choice(actions)
        else:
            # Exploitation: choose the best action based on current Q-values
            state_actions = {action: q for action, q in self.q_table.items() if action[0] == state}
            return max(state_actions, key=state_actions.get, default=None)

    def update_q_value(self, state, action, reward, next_state):
        next_max = float('-inf')  # Initialize to negative infinity to ensure any real Q-value is higher
        if next_state in self.graph:
            for neighbor in self.graph[next_state]:
                for mode_key in self.graph[next_state][neighbor]:
                    action_key = (next_state, neighbor, mode_key)
                    if action_key in self.q_table:
                        next_max = max(next_max, self.q_table[action_key])

        # If there were no actions from next_state
        if next_max == float('-inf'):
            next_max = 0
        # Update the Q-value for the current action
        self.q_table[action] = self.q_table.get(action, 0) + \
                               self.alpha * (reward + self.gamma * next_max - self.q_table.get(action, 0))

    def update_epsilon(self):
        # Decay epsilon to reduce exploration over time
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def learn(self, start, destination, episodes=500, progress_check_interval=100):
        for episode in range(1, episodes + 1):
            current_state = start
            done = False
            while not done:
                action = self.choose_action(current_state)
                _, next_state, mode = action
                reward = -self.graph[current_state][next_state][mode][
                    'weight']
                self.update_q_value(current_state, action, reward, next_state)
                current_state = next_state
                if current_state == destination:
                    done = True
            # Update epsilon after each episode
            self.update_epsilon()
            if episode % progress_check_interval == 0:
                print(f"Episode {episode}/{episodes} completed")

    def print_optimal_route(self, start, destination):
        current_state = start
        visited_states = {start}  # Initialize with the start state
        optimal_route = [start]
        modes_used = []
        time_cost = 0

        print("Optimal Route and Modes:")

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

            time_cost = time_cost + self.graph.get_edge_data(current_state, next_state, key=mode)['weight']
            optimal_route.append(next_state)
            modes_used.append(mode)
            visited_states.add(next_state)  # Mark the next_state as visited
            current_state = next_state

        # Print the optimal route and modes used
        for i in range(len(optimal_route) - 1):
            print(f"{optimal_route[i]} -> {optimal_route[i + 1]} (Mode: {modes_used[i]})")
        print(f"Destination reached: {destination}\n Time: {time_cost}")
        return time_cost


if __name__ == '__main__':
    net_xml_path = 'DCC.net.xml'
    source_edge = '3789374#3'
    target_edge = "-361450282"
    start_mode = 'walking'
    db_path = 'test_new.db'
    user = User(60, False, 0, 20)

    optimizer = Optimization(net_xml_path, user, db_path, source_edge, target_edge)
    graph = optimizer.new_graph
    agent = MultiModalQLearningAgent(graph)
    start_time = time.time()
    agent.learn(source_edge, target_edge)
    end_time = time.time()
    print(f"Total time: {end_time - start_time} seconds")
    time = agent.print_optimal_route(source_edge, target_edge)
