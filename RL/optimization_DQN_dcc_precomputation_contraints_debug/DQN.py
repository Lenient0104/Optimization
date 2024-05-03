import time

import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

from matplotlib import pyplot as plt

from user_info import User
from optimization import Optimization


class Environment:
    def __init__(self, graph, source, destination, initial_energy=100):
        self.graph = graph
        self.source = source
        self.destination = destination
        self.current_node = source
        self.remaining_energy = initial_energy
        self.initial_energy = initial_energy
        self.last_mode = 'walking'
        self.reset()

    def reset(self):
        self.current_node = self.source
        self.remaining_energy = self.initial_energy  # Reset remaining energy
        self.last_mode = 'walking'  # Reset the last mode used
        return self._get_state()

    def _get_state(self):
        current_node_index = list(self.graph.nodes).index(self.current_node)
        state = np.array([current_node_index])
        return state

    def step(self, action_index):
        # Fetch all possible actions from the current node, including their mode as a key for lookup
        possible_actions = [(neighbor, key, data) for neighbor in self.graph.neighbors(self.current_node)
                            for key, data in self.graph[self.current_node][neighbor].items()]


        if not possible_actions or action_index >= len(possible_actions):
            # Invalid action chosen or no actions available; penalize heavily
            info = {'current_node': self.current_node, 'mode': self.last_mode, 'action_taken': 'None'}
            return self._get_state(), -10000, False, info  # Now includes info


        # Select the action based on the action_index
        selected_action = possible_actions[action_index]
        next_node, mode, edge_data = selected_action

        # Fetch edge data using the current node, next node, and mode
        edge_data = self.graph[self.current_node][next_node][mode]
        distance = edge_data['distance']
        time_cost = edge_data['weight']  # Assuming 'weight' holds the time cost

        # Calculate energy consumption
        energy_consumed = self.calculate_energy_comsumption(mode, distance)

        # Check for mode transfer and refill energy if there's a change in mode
        if mode != self.last_mode and self.last_mode is not None:
            self.remaining_energy = 100  # Refill energy on mode transfer

        # Check if the action is feasible within the energy constraint
        if self.remaining_energy - energy_consumed < 0:
            # Action not feasible due to energy constraint, so don't change mode
            info = {'current_node': self.current_node, 'mode': self.last_mode, 'action_taken': 'Insufficient energy'}
            return self._get_state(), -10000, False, info  # Now includes info


        self.last_mode = mode  # Update the last mode used
        # Update energy and current node as the action is feasible
        self.remaining_energy -= energy_consumed
        self.current_node = next_node

        # The reward is now the negative of the time cost
        reward = -time_cost

        # Check if the destination has been reached
        done = self.current_node == self.destination



        info = {
            'current_node': self.current_node,  # Assuming self.current_node tracks the current position
            'mode': self.last_mode,
            'action_taken': action  # Including the action taken can be useful for debugging
            # Add any other info you might find relevant
        }

        return self._get_state(), reward, done, info

    # def update_state_energy_and_check_done(self, next_node, energy_consumed):
    #     # Update the environment's current node and remaining energy
    #     self.current_node = next_node
    #     self.remaining_energy -= energy_consumed
    #
    #     # Check if the destination is reached or energy is depleted
    #     if next_node == self.destination:
    #         return 100, True  # Reward for reaching the destination
    #     elif self.remaining_energy <= 0:
    #         return -100, True  # Penalty for depleting energy
    #
    #     return -1, False  # Standard step penalty to encourage efficiency

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


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995,
                 min_epsilon=0.01, buffer_size=10000, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Experiences memory
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.lr = lr

        self.model = DQN(state_dim, action_dim, hidden_dim)
        self.target_model = DQN(state_dim, action_dim, hidden_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()  # Set target_model to evaluation mode

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        # recording
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Explore
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)
        # Exploit
        state = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor and add batch dimension
        with torch.no_grad():  # Disable gradient calculation for inference
            q_values = self.model(state)
        return np.argmax(q_values.cpu().numpy())  # Choose the action with the highest Q-value

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convert lists of tuples to NumPy arrays with the correct shape
        states = np.array(states, dtype=np.float32)  # states are shaped as (batch_size, 2)
        next_states = np.array(next_states, dtype=np.float32)

        # Conversion to tensors
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions).view(-1, 1)  # Ensure actions are properly shaped for gather
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        current_q_values = self.model(states).gather(1, actions).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(current_q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


# Initialize your environment
net_xml_path = 'DCC.net.xml'
source_edge = '361450282'
destination_edge = "-110407380#1"
user = User(60, False, 0, 20)
optimizer = Optimization(net_xml_path, user, 'test_new.db', source_edge, destination_edge)
graph = optimizer.new_graph
env = Environment(graph, source_edge, destination_edge)

# Define state and action dimensions
state_dim = 1  # For example: [current_node_index]
action_dim = max(
    len(list(graph.out_edges(node))) for node in graph.nodes)  # Max number of possible actions from any node


# Define episodes
episodes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
test_size = 30

# Lists to store average rewards and execution times
avg_rewards = []
avg_exe_times = []

# Perform experiments
for episode_count in episodes:
    agent = DQNAgent(state_dim, action_dim)
    print(episode_count)
    total_rewards = []
    exe_times = []
    for _ in range(test_size):  # Run 30 times for each episode count
        start_time = time.time()
        best_total_reward = float('-inf')

        for episode in range(episode_count):
            state = env.reset()
            total_reward = 0
            done = False
            route = [source_edge]  # Initialize an empty route
            modes_used = ['walking']  # Track modes used for each step in the route

            while not done:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)

                # Append current node and mode to their respective lists
                route.append(info['current_node'])
                modes_used.append(info['mode'])

                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if done:
                    agent.update_target_model()

            if total_reward > best_total_reward and route[-1] == destination_edge:
                best_total_reward = total_reward

            if len(agent.memory) > 32:
                agent.replay()

        total_rewards.append(best_total_reward)
        exe_times.append(time.time() - start_time)

    avg_rewards.append(-np.mean(total_rewards))  # Taking the negative mean of rewards
    avg_exe_times.append(np.mean(exe_times))

# Creating subplots with 2 rows and 1 column
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plotting execution time against number of episodes
ax1.plot(episodes, avg_exe_times, 'ro-', label='Execution Time')
ax1.set_xlabel('Number of Episodes', fontsize=16)
ax1.set_ylabel('Execution Time (seconds)', fontsize=16)
ax1.set_title('DQN Performance: Execution Time vs. Number of Episodes', fontsize=18)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.legend()
ax1.grid(True)

# Plotting average rewards against number of episodes
ax2.plot(episodes, avg_rewards, 'bs-', label='Average Reward')
ax2.axhline(y=3084.136579657965, color='g', linestyle='--', label='Reference Value')
ax2.set_xlabel('Number of Episodes', fontsize=16)
ax2.set_ylabel('Average time costs (s)', fontsize=16)
ax2.set_title('DQN Performance: Average Reward vs. Number of Episodes', fontsize=18)
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

