import time
import csv

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
        self.visited_nodes = set()
        self.reset()

    def get_possible_actions(self, node):
        return [(neighbor, key, data) for neighbor in self.graph.neighbors(node)
                for key, data in self.graph[node][neighbor].items()]

    def reset(self):
        self.visited_nodes = set()
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

        # Select the action based on the action_index
        selected_action = possible_actions[action_index]

        next_node, mode, edge_data = selected_action

        if next_node == self.current_node or next_node in self.visited_nodes:
            # print('loop')
            reward = -100000
            info = {'current_node': self.current_node, 'mode': mode, 'action_taken': 'Loop detected'}
            return self._get_state(), reward, False, info

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
            # print("no enough energy")
            # Action not feasible due to energy constraint, so don't change mode
            info = {'current_node': self.current_node, 'mode': self.last_mode, 'action_taken': 'Insufficient energy'}
            return self._get_state(), -100000, False, info  # Now includes info

        self.last_mode = mode  # Update the last mode used
        # Update energy and current node as the action is feasible
        self.remaining_energy -= energy_consumed
        self.current_node = next_node
        self.visited_nodes.add(next_node)
        # The reward is now the negative of the time cost
        reward = -time_cost


        # Check if the destination has been reached
        done = self.current_node == self.destination


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
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=0.5, gamma=0.95, epsilon=1.0, epsilon_decay=0.995,
                 min_epsilon=0.01, buffer_size=100000, batch_size=64):
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

    def act(self, state, num_actions, test=False):
        if test:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state)
            q_values_array = q_values.cpu().numpy()[0][:num_actions]
            action = np.argmax(q_values_array)
            return action
        else:
            if np.random.rand() < self.epsilon:
                action = random.randrange(num_actions)
            else:
                state = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_values = self.model(state)
                q_values_array = q_values.cpu().numpy()[0][:num_actions]
                action = np.argmax(q_values_array)
            return action

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


# after all the training finished
def infer_best_route(agent, env, max_steps=1000):
    state = env.reset()
    best_route = [env.current_node]
    best_modes = [env.last_mode]
    total_time_cost = 0
    steps = 0
    find = False

    while steps < max_steps:
        possible_actions = [(neighbor, key, data) for neighbor in env.graph.neighbors(env.current_node)
                            for key, data in env.graph[env.current_node][neighbor].items()]
        num_available_actions = len(possible_actions)
        if num_available_actions == 0:
            break

        action = agent.act(state, num_available_actions, test=True)

        next_state, reward, done, info = env.step(action)

        if done:
            best_route.append(info['action_taken'][0])
            best_modes.append(info['mode'])
            total_time_cost -= reward

            print(best_route)
            print(best_modes)
            print(total_time_cost)
            return best_route, best_modes, total_time_cost, True

        best_route.append(info['action_taken'][0])
        best_modes.append(info['mode'])
        total_time_cost -= reward

        state = next_state
        steps += 1
    # print('best route: ', best_route)
    return best_route, best_modes, total_time_cost, find


# Initialize your environment
net_xml_path = 'DCC.net.xml'
source_edge = '361450282'
destination_edge = "-110407380#1"
user = User(60, True, 0, 20)
optimizer = Optimization(net_xml_path, user, 'test_new.db', source_edge, destination_edge)
graph = optimizer.new_graph
env = Environment(graph, source_edge, destination_edge)

# Define state and action dimensions
state_dim = 1  # For example: [current_node_index]
action_dim = max(
    len(list(graph.out_edges(node))) for node in graph.nodes)  # Max number of possible actions from any node

# Define episodes
episodes = [400]
test_size = 30
done = False

all_DQN_exe_times = []
all_DQN_times = []
all_successful_tests = []  # 新增列表用于存储每个episode的成功测试次数

# Create a new CSV file to store the results
with open('results/DQN_experiment_results_episode400_20stations.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # Write the header of the CSV file
    writer.writerow(['Experiment ID', 'Episode', 'Execution Time (seconds)', 'Time Cost', 'Find'])

    # Perform experiments
    for episode_count in episodes:
        episode_exe_times = []
        episode_times = []
        successful_tests = 0
        print("Episode number: ", episode_count)
        for test_index in range(test_size):  # Run 30 times for each episode count
            print("test_index", test_index)
            agent = DQNAgent(state_dim, action_dim)
            start_time = time.time()
            update_frequency = 10
            rewards = []

            for episode in range(episode_count):
                total_reward = 0  # 初始化该回合的总奖励
                rewards = []
                state = env.reset()
                route = [env.current_node]
                done = False
                while not done:
                    possible_actions = [
                        (neighbor, key, data)
                        for neighbor in env.graph.neighbors(env.current_node)
                        for key, data in env.graph[env.current_node][neighbor].items()
                    ]
                    num_available_actions = len(possible_actions)
                    action = agent.act(state, num_available_actions, test=False)
                    next_state, reward, done, info = env.step(action)
                    route.append(env.current_node)
                    agent.remember(state, action, reward, next_state, done)

                    if next_state == state:
                        break  # 防止在相同状态循环
                    state = next_state
                    # print(total_reward)
                    total_reward += reward  # 更新本回合总奖励
                    rewards.append(total_reward)  # 记录该回合结束时的总奖励

                    if len(agent.memory) > 16:
                        agent.replay()
                if (episode + 1) % update_frequency == 0:
                    agent.update_target_model()

                # if done:
                #     steps = list(range(len(rewards)))  # 创建一个与奖励列表相同长度的回合数列表
                #     # print(rewards)
                #     plt.figure(figsize=(10, 5))
                #     plt.plot(steps, rewards)
                #     plt.xlabel('Move Steps from the Start Edge')  # 横轴标签
                #     plt.ylabel('Total Reward till Current Move Step')  # 纵轴标签
                #     plt.title('DQN: Reward Progression over Move Steps')  # 图表标题
                #     plt.grid(True)  # 显示网格
                #     plt.show()  # 显示图表


            end_time = time.time()
            execution_time = end_time - start_time
            best_route, best_modes, total_time_cost, find = infer_best_route(agent, env)

            # Write each test's result to the CSV file
            experiment_id = f"{episode_count}-{test_index}"
            writer.writerow([experiment_id, episode_count, execution_time, total_time_cost, find])

            if find:
                episode_exe_times.append(execution_time)
                episode_times.append(total_time_cost)
                successful_tests += 1


        all_DQN_exe_times.append(episode_exe_times)
        all_DQN_times.append(episode_times)


# 定义 boxplot 的样式
boxprops = dict(linestyle='-', linewidth=1.5, color='black', facecolor='cornflowerblue')
medianprops = dict(linestyle='-', linewidth=1.5, color='darkblue')
flierprops = dict(marker='o', color='black', alpha=0.5)

# 找到最大长度
max_length = max(max(len(times) for times in all_DQN_exe_times), max(len(times) for times in all_DQN_times))

# 填充所有数组到最大长度
filled_exe_times = [times + [np.mean(times)] * (max_length - len(times)) for times in all_DQN_exe_times]
filled_times = [times + [np.mean(times)] * (max_length - len(times)) for times in all_DQN_times]

# 创建画布和子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# 绘制执行时间的箱型图
ax1.boxplot(filled_exe_times, positions=episodes, widths=45, boxprops=boxprops,
            medianprops=medianprops, flierprops=flierprops, patch_artist=True)
ax1.set_xticks(episodes)
ax1.set_xticklabels(episodes, rotation=45)
ax1.set_xlabel('Number of Episodes', fontsize=16)
ax1.set_ylabel('Execution Time (seconds)', fontsize=16)
ax1.set_title('DQN Performance: Execution Time vs. Number of Episodes', fontsize=18)
ax1.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)

# 绘制时间成本的箱型图
ax2.boxplot(filled_times, positions=episodes, widths=45, boxprops=boxprops,
            medianprops=medianprops, flierprops=flierprops, patch_artist=True)
ax2.set_xticks(episodes)
ax2.set_xticklabels(episodes, rotation=45)
ax2.set_xlabel('Number of Episodes', fontsize=16)
ax2.set_ylabel('Time Cost (seconds)', fontsize=16)
ax2.set_title('DQN Performance: Time Cost vs. Number of Episodes', fontsize=18)
ax2.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)

plt.tight_layout()
plt.show()