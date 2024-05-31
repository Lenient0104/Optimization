import sys
import csv
import time
import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

from matplotlib import pyplot as plt


class Environment:
    def __init__(self, graph, source, destination, energy_rate, initial_energy=100):
        self.graph = graph
        self.source = source
        self.total_time_cost = 0
        self.destination = destination
        self.current_node = source
        self.remaining_energy = initial_energy
        self.initial_energy = initial_energy * energy_rate
        self.last_mode = 'walking'
        self.visited_nodes = set()
        self.reset()
        self.steps = 0

    def get_possible_actions(self, node):
        return [(neighbor, key, data) for neighbor in self.graph.neighbors(node)
                for key, data in self.graph[node][neighbor].items()]

    def reset(self):
        self.visited_nodes = set()
        self.current_node = self.source
        self.total_time_cost = 0
        self.steps = 0
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

        # Select the action based on the action_index
        selected_action = possible_actions[action_index]
        #
        # print(selected_action)

        next_node, mode, edge_data = selected_action

        if next_node == self.current_node or next_node in self.visited_nodes:
            # print('loop')
            reward = - 1000
            self.current_node = next_node
            info = {'current_node': self.current_node, 'mode': mode, 'action_taken': 'Loop detected'}
            return self._get_state(), reward, False, info

        # Fetch edge data using the current node, next node, and mode
        edge_data = self.graph[self.current_node][next_node][mode]
        distance = edge_data['distance']
        time_cost = edge_data['weight']  # Assuming 'weight' holds the time cost

        energy_consumed = self.calculate_energy_comsumption(mode, distance)

        # Check for mode transfer and refill energy if there's a change in mode
        if mode != self.last_mode and self.last_mode is not None:
            self.remaining_energy = self.initial_energy
            # print(self.remaining_energy, self.initial_energy)# Refill energy on mode transfer

        # Check if the action is feasible within the energy constraint
        if self.remaining_energy - energy_consumed < 0:
            # print("no enough energy")
            self.current_node = next_node
            # Action not feasible due to energy constraint, so don't change mode
            info = {'current_node': self.current_node, 'mode': self.last_mode, 'action_taken': 'Insufficient energy'}
            return self._get_state(), -1000, False, info  # Now includes info

        self.last_mode = mode  # Update the last mode used
        # Update energy and current node as the action is feasible
        self.remaining_energy -= energy_consumed
        current_state = self._get_state()
        self.visited_nodes.add(next_node)
        # The reward is now the negative of the time cost divided by distance to consider path length
        reward = 1 / time_cost
        self.steps += 1
        self.current_node = next_node
        self.total_time_cost += time_cost

        # Check if the destination has been reached
        done = self.current_node == self.destination

        if done and self.steps >= 3:
            reward = 1000000 / self.total_time_cost
            # print(self.total_time_cost)
        info = {
            'current_node': self.current_node,
            'mode': self.last_mode,
            'action_taken': selected_action  # Assuming `selected_action` is a descriptive action representation
        }

        return self._get_state(), reward, done, info

    def calculate_energy_comsumption(self, current_mode, distance):
        if current_mode == 'walking':
            return 0
        # Define vehicle efficiency in Wh per meter (converted from Wh per km)
        vehicle_efficiency = {'e_bike_1': 20 / 1000, 'e_scooter_1': 25 / 1000, 'e_car': 150 / 1000}
        # battery_capacity = {'e_bike_1': 500, 'e_scooter_1': 250, 'e_car': 50000}
        battery_capacity = {'e_bike_1': 500, 'e_scooter_1': 250, 'e_car': 5000}
        energy_consumed = vehicle_efficiency[current_mode] * distance
        # Calculate the required SoC (%) for the distance traveled
        delta_soc = (energy_consumed / battery_capacity[current_mode]) * 100

        return delta_soc


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        # # 初始化最后一层的权重和偏置以使初始Q值为-200
        # self.net[-1].weight.data.fill_(0)
        # self.net[-1].bias.data.fill_(0)

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=512, lr=0.001, gamma=0.8, epsilon=1.0, epsilon_decay=0.999,
                 min_epsilon=0.01, buffer_size=5000, batch_size=256, n_steps=3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.n_steps = n_steps  # 这里定义n步前瞻

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.lr = lr

        self.model = DQN(state_dim, action_dim, hidden_dim)
        self.target_model = DQN(state_dim, action_dim, hidden_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        self.loss_history = []

    def remember(self, state, action, reward, next_state, steps, done):
        # recording
        self.memory.append((state, action, reward, next_state, steps, done))

    def act(self, state, num_actions, start, test=False):

        if test:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state)
            q_values_array = q_values.cpu().numpy()[0][:num_actions]
            action = np.argmax(q_values_array)
            max_q_value = np.max(q_values_array)
            # print("test max q value:", max_q_value)
            return action
        else:
            if np.random.rand() < self.epsilon:
                action = random.randrange(num_actions)
                state = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_values = self.model(state)
                q_values_array = q_values.cpu().numpy()[0][:num_actions]
                action_q_value = q_values_array[action]
                # print("random q value:", action_q_value)
            else:
                print('=========max========')
                state = torch.FloatTensor(state).unsqueeze(0)
                # print('state:', state)
                with torch.no_grad():
                    q_values = self.model(state)
                q_values_array = q_values.cpu().numpy()[0][:num_actions]
                # print(q_values_array)
                action = np.argmax(q_values_array)
                max_q_value = np.max(q_values_array)
                print('max q value', max_q_value)

                # if start:
                #     print("training max q value:", max_q_value)
            return action

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for sample in minibatch:
            state, action, reward, next_state, _, done = sample
            # 在这里处理 n 步逻辑
            total_return = reward
            current_state = state
            current_action = action
            gamma_pow = 1
            transition_idx = list(self.memory).index(sample)

            # 检查连续性并计算累计奖励
            for _ in range(1, self.n_steps):
                if transition_idx + 1 < len(self.memory):
                    next_sample = self.memory[transition_idx + 1]
                    if not next_sample[4]:  # 如果没有结束
                        gamma_pow *= self.gamma
                        total_return += gamma_pow * next_sample[2]
                        next_state = next_sample[3]
                        transition_idx += 1
                    else:
                        break

            if not done:
                total_return += (self.gamma ** self.n_steps) * np.max(
                    self.model(torch.FloatTensor(next_state)).detach().numpy())

            current_q_value = self.model(torch.FloatTensor(current_state))[current_action]
            loss = self.loss_fn(current_q_value, torch.tensor(total_return).float())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 更新 epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


# after all the training finished
def infer_best_route(agent, optimizer, env, max_steps=1000):
    state = env.reset()
    print(state)
    best_route = [env.current_node]
    best_modes = []
    total_time_cost = 0
    steps = 0
    find = False
    start = False

    while steps < max_steps:
        current_node = env.current_node
        possible_actions = [(neighbor, key, data) for neighbor in env.graph.neighbors(env.current_node)
                            for key, data in env.graph[env.current_node][neighbor].items()]
        # print(possible_actions)
        num_available_actions = len(possible_actions)
        if num_available_actions == 0:
            break

        action = agent.act(state, num_available_actions, start, test=True)

        next_state, reward, done, info = env.step(action)
        print(next_state)
        if info['action_taken'] == 'Loop detected' or current_node == env.current_node:
            print('==========')
            print(info)
            print('=============')
            return best_route, best_modes, total_time_cost, find
        # print("================")
        # print(current_node)
        # print(env.current_node)
        # print(optimizer.new_graph[current_node])
        # print(optimizer.new_graph[current_node][env.current_node])
        # print("==================")

        distance = optimizer.new_graph[current_node][env.current_node][info['mode']]['distance']
        time_cost = optimizer.new_graph[current_node][env.current_node][info['mode']]['weight']
        # print(distance, time_cost)

        if info['action_taken'][0] == env.destination:
            best_route.append(info['action_taken'][0])
            best_modes.append(info['mode'])

            total_time_cost += time_cost

            # print(best_route)
            # print(best_modes)
            # print(total_time_cost)
            return best_route, best_modes, total_time_cost, True

        best_route.append(info['action_taken'][0])
        best_modes.append(info['mode'])
        total_time_cost += time_cost

        state = next_state
        steps += 1
    # print('best route: ', best_route)
    return best_route, best_modes, total_time_cost, find


def plot_loss(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Loss over time')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()


def run_dqn(optimizer, source_edge, target_edge, episode_number, energy_rate):
    all_DQN_exe_times = []
    all_DQN_times = []
    all_successful_tests = []

    episode_exe_times = []
    episode_times = []
    successful_tests = 0

    state_dim = 1  # [current_node_index]
    action_dim = max(
        len(list(optimizer.new_graph.out_edges(node))) for node in
        optimizer.new_graph.nodes)  # Max number of possible actions from any node

    env = Environment(optimizer.new_graph, source_edge, target_edge, energy_rate)
    agent = DQNAgent(state_dim, action_dim)
    start_time = time.time()
    update_frequency = 100
    results = []

    for episode in range(episode_number):
        print("-----------------------------------------Episode-------------------------------------", episode)
        total_reward = 0
        env.total_time_cost = 0
        state = env.reset()
        route = [env.current_node]
        done = False
        rewards_count = []
        modes = []
        start = True

        while not done:
            possible_actions = [
                (neighbor, key, data)
                for neighbor in env.graph.neighbors(env.current_node)
                for key, data in env.graph[env.current_node][neighbor].items()
            ]
            num_available_actions = len(possible_actions)
            action = agent.act(state, num_available_actions, start, test=False)
            start = False
            _, mode, _ = possible_actions[action]
            next_state, reward, done, info = env.step(action)
            # print(reward)
            # print(info)
            route.append(env.current_node)
            modes.append(mode)
            rewards_count.append(reward)
            # env.total_time_cost -= reward
            agent.remember(state, action, reward, next_state, env.steps, done)
            print(state, action, reward, next_state, done)
            total_size = sum(sys.getsizeof(x) for x in (state, action, reward, next_state, done))

            # print("Approximate size of the tuple in bytes:", total_size)

            if reward < 0.000001:
                break  # Avoid looping in the same state
            state = next_state
            total_reward += reward
            #
            if done:
                print(env.total_time_cost)
                results.append(env.total_time_cost)

            agent.replay()

        if (episode + 1) % update_frequency == 0:
            agent.update_target_model()
            print('Model updated and loss plotted.')
        # plot_loss(agent.loss_history)
        # print(agent.loss_history)

    memory = agent.memory
    end_time = time.time()
    execution_time = end_time - start_time
    best_route, best_modes, total_time_cost, find = infer_best_route(agent, optimizer, env)
    total_time_cost = total_time_cost + optimizer.edge_map[target_edge]['length'] / 1.5

    if find:
        episode_exe_times.append(execution_time)
        episode_times.append(total_time_cost)
        successful_tests += 1

    all_DQN_exe_times.append(episode_exe_times)
    all_DQN_times.append(episode_times)

    plt.plot(results)
    plt.show()

    return best_route, best_modes, total_time_cost, execution_time, find