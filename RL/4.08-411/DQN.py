import time
import csv

import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import numpy as np


from matplotlib import pyplot as plt
from collections import deque

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
        self.action_history = deque(maxlen=3)  # 初始化行动历史
        self.max_possible_actions = max(len(self.get_possible_actions(node)) for node in self.graph.nodes)
        self.reset()  # 现在可以安全地调用reset方法


    def get_possible_actions(self, node):
        return [(neighbor, key, data) for neighbor in self.graph.neighbors(node)
                for key, data in self.graph[node][neighbor].items()]

    def reset(self):
        self.visited_nodes = {self.current_node}
        self.current_node = self.source
        self.remaining_energy = self.initial_energy  # Reset remaining energy
        self.last_mode = 'walking'  # Reset the last mode used
        return self._get_state()

    def _get_state(self):
        current_node_index = list(self.graph.nodes).index(self.current_node)
        # 使用最大可能动作数来标准化状态长度
        history_encoded = [0] * (self.max_possible_actions * 3)

        for i, action in enumerate(self.action_history):
            index = i * self.max_possible_actions + action % self.max_possible_actions  # 使用取余确保索引有效
            history_encoded[index] = 1

        state = [current_node_index] + history_encoded
        return np.array(state)

    def step(self, action_index):

        new_energy = 100  # 切换模式时能量重置
        possible_actions = [(neighbor, key, data) for neighbor in self.graph.neighbors(self.current_node)
                            for key, data in self.graph[self.current_node][neighbor].items()]
        # print("Number of possible actions:", len(possible_actions))

        # 处理非法行动
        if action_index >= len(possible_actions):
            return self._get_state(), -10000, False, {'current_node': self.current_node,
                                                      'action_taken': 'Invalid action'}

        selected_action = possible_actions[action_index]
        next_node, mode, edge_data = selected_action

        distance = edge_data['distance']
        time_cost = edge_data['weight']  # 假设'weight'代表行走时间成本
        energy_consumed_new = self.calculate_energy_comsumption(mode, distance)
        energy_consumed_current = self.calculate_energy_comsumption(self.last_mode, distance)

        # 检查能量是否足够进行当前选择的移动方式
        if mode == self.last_mode and self.remaining_energy - energy_consumed_current < 0:
            return self._get_state(), -100000, False, {'current_node': self.current_node, 'mode': self.last_mode,
                                                       'action_taken': 'Insufficient energy'}
        elif mode != self.last_mode and new_energy - energy_consumed_new < 0:
            return self._get_state(), -100000, False, {'current_node': self.current_node, 'mode': mode,
                                                       'action_taken': 'Insufficient energy for new mode'}

        # 更新当前节点、剩余能量、模式
        self.current_node = next_node
        self.last_mode = mode
        if mode == self.last_mode:
            self.remaining_energy -= energy_consumed_current
        else:
            self.remaining_energy = new_energy - energy_consumed_new  # 模式变更时重置并减少能量

        # 记录访问过的节点，奖励新探索
        if next_node in self.visited_nodes:
            reward = -10000000000  # 重访节点的重罚
            print(f"Revisited node penalty applied: {next_node}")
        else:
            reward = 1000 - time_cost  # 探索新节点的奖励，减去时间成本
            self.visited_nodes.add(next_node)
            print(f"Visited new node: {next_node}, reward updated")

        done = self.current_node == self.destination
        info = {'current_node': self.current_node, 'mode': self.last_mode, 'action_taken': selected_action}

        # 更新行动历史
        self.action_history.append(action_index)  # 此处应确保你的类定义中包含了action_history属性

        return self._get_state(), reward, done, info

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
    def __init__(self, state_dim, action_dim, hidden_dim=256):  # 增加hidden_dim以提供更大的网络容量
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.fc4 = nn.Linear(hidden_dim * 2, action_dim)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        x = self.fc4(x)  # 没有激活函数，直接输出Q值
        return x



class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=0.001, gamma=0.95, initial_epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, buffer_size=100000, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay  # 确保这个属性被正确初始化
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.lr = lr

        self.model = DQN(state_dim, action_dim, hidden_dim)
        self.target_model = DQN(state_dim, action_dim, hidden_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        # UCB所需的其他属性
        self.action_selection_count = np.zeros(action_dim)
        self.total_action_count = 0

    def select_action_ucb(self, state, num_available_actions):
        c = 2  # UCB的探索参数
        q_values = self.model(torch.FloatTensor(state).unsqueeze(0)).detach().numpy()[0][:num_available_actions]
        ucb_values = q_values + c * np.sqrt(
            np.log(self.total_action_count + 1) / (self.action_selection_count[:num_available_actions] + 1))
        action = np.argmax(ucb_values)
        self.action_selection_count[action] += 1
        self.total_action_count += 1
        return action

    def remember(self, state, action, reward, next_state, done):
        # recording
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, num_available_actions=None, test=False):
        if num_available_actions is None:
            num_available_actions = self.action_dim
        if test:
            with torch.no_grad():
                q_values = self.model(torch.FloatTensor(state).unsqueeze(0))
            action = np.argmax(q_values.cpu().numpy()[:num_available_actions])
        else:
            action = self.select_action_ucb(state, num_available_actions)
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

    while steps < max_steps:
        possible_actions = [(neighbor, key, data) for neighbor in env.graph.neighbors(env.current_node)
                            for key, data in env.graph[env.current_node][neighbor].items()]
        num_available_actions = len(possible_actions)
        print(f"Step {steps}, Current Node: {env.current_node}, Possible Actions: {num_available_actions}")

        if num_available_actions == 0:
            print("No available actions.")
            break

        action = agent.act(state, num_available_actions, test=True)
        print(f"Selected Action: {action}")

        next_state, reward, done, info = env.step(action)
        print(f"Next State: {next_state}, Reward: {reward}, Done: {done}")

        total_time_cost -= reward
        if done:
            print("Destination found.")
            best_route.append(info['current_node'])
            best_modes.append(env.last_mode)
            break

        if np.array_equal(next_state, state):
            print("No progress, stuck in loop.")
            break

        state = next_state
        steps += 1
        best_route.append(info['current_node'])
        best_modes.append(env.last_mode)

    if steps >= max_steps:
        print("Max steps reached without finding destination.")

    print('Best Route:', best_route)
    print('Modes Used:', best_modes)
    print('Total Time Cost:', total_time_cost)

    return best_route, best_modes, total_time_cost


# Initialize your environment
net_xml_path = 'DCC.net.xml'
source_edge = '361450282'
destination_edge = "-110407380#1"
user = User(60, False, 0, 20)
optimizer = Optimization(net_xml_path, user, 'test_new.db', source_edge, destination_edge)
graph = optimizer.new_graph
env = Environment(graph, source_edge, destination_edge)
state_dim = 1 + env.max_possible_actions * 3
action_dim = max(len(list(graph.out_edges(node))) for node in graph.nodes)
agent = DQNAgent(state_dim, action_dim)


# Define episodes
episodes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
test_size = 30
done = False

all_DQN_exe_times = []
all_DQN_times = []

# Perform experiments
for episode_count in episodes:
    episode_exe_times = []
    episode_times = []
    print("Episode number: ", episode_count)

    for test_index in range(test_size):  # Run 30 times for each episode count
        print("test_index", test_index)
        env.reset()  # 只重置环境状态
        start_time = time.time()

        for episode in range(episode_count):
            state = env.reset()
            done = False
            while not done:
                current_node = env.current_node
                possible_actions = env.get_possible_actions(env.current_node)
                num_available_actions = len(possible_actions)
                action = agent.act(state, num_available_actions, test=False)
                next_state, reward, done, info = env.step(action)
                next_node = info['current_node']
                agent.remember(state, action, reward, next_state, done)

                # Debug output
                print(f"Current state: {state}, Action: {action}, Reward: {reward}, Next state: {next_state}")

                if current_node == next_node:
                    print("Stuck in loop, breaking...")
                    break
                state = next_state

                # Perform learning only if enough memories are available
                if len(agent.memory) > agent.batch_size:
                    agent.replay()

            agent.update_target_model()

        # 结束一次完整的测试
        end_time = time.time()
        execution_time = end_time - start_time
        episode_exe_times.append(execution_time)
        best_route, best_modes, total_time_cost = infer_best_route(agent, env)
        episode_times.append(total_time_cost)

    all_DQN_exe_times.append(episode_exe_times)
    all_DQN_times.append(episode_times)

# Optionally, save the data to a CSV file
with open('DQN_results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Episode', 'Execution Time (seconds)', 'Time Cost'])

    for i, episode in enumerate(episodes):
        for j in range(test_size):
            writer.writerow([episode, all_DQN_exe_times[i][j], all_DQN_times[i][j]])

# Custom boxplot appearance
boxprops = dict(linestyle='-', linewidth=1.5, color='black', facecolor='cornflowerblue')
medianprops = dict(linestyle='-', linewidth=1.5, color='darkblue')
flierprops = dict(marker='o', color='black', alpha=0.5)

# Creating subplots with 2 rows and 1 column
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Boxplot for Execution Time
ax1.boxplot(all_DQN_exe_times, positions=episodes, widths=35, boxprops=boxprops,
            medianprops=medianprops, flierprops=flierprops, patch_artist=True)
ax1.set_xticklabels(episodes, rotation=45, ha='right')
ax1.set_xlabel('Number of Episodes', fontsize=16)
ax1.set_ylabel('Execution Time (seconds)', fontsize=16)
ax1.set_title('DQN Performance: Execution Time vs. Number of Episodes', fontsize=18)
ax1.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)
ax1.legend(['Execution Time'])

# Boxplot for Time Cost
ax2.boxplot(all_DQN_times, positions=episodes, widths=35, boxprops=boxprops,
            medianprops=medianprops, flierprops=flierprops, patch_artist=True)
ax2.set_xticklabels(episodes, rotation=45, ha='right')
ax2.set_xlabel('Number of Episodes', fontsize=16)
ax2.set_ylabel('Travel Time Cost (seconds)', fontsize=16)
ax2.set_title('DQN Performance: Travel Time Cost vs. Number of Episodes', fontsize=18)
ax2.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)
ax2.legend(['Travel Time Cost'])

plt.tight_layout()
plt.show()
