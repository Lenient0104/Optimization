import time
import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque


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
        # The reward is now the negative of the time cost divided by distance to consider path length
        reward = - time_cost

        # Check if the destination has been reached
        done = self.current_node == self.destination

        # if done:
        #     reward = 100
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
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=512, lr=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.999999,
                 min_epsilon=0.05, buffer_size=1000000, batch_size=16):
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
                # print("random")
            else:
                # print("max")
                state = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_values = self.model(state)
                q_values_array = q_values.cpu().numpy()[0][:num_actions]
                action = np.argmax(q_values_array)
            return action

    def replay(self, n_steps=3):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for state, action, reward, next_state, done in minibatch:
            total_reward = reward
            current_state = state
            current_next_state = next_state
            current_done = done

            # Start accumulating future rewards up to n_steps
            future_steps = min(n_steps, self.batch_size - 1)  # Limit the number of steps to batch size
            for _ in range(future_steps):
                if current_done:
                    break
                # Get next experience from memory (simulate future steps)
                future_action = np.random.choice(range(len(self.memory)))  # Randomly select a future action
                _, _, future_reward, future_next_state, future_done = self.memory[future_action]

                total_reward += (self.gamma ** future_steps) * future_reward
                current_next_state = future_next_state
                current_done = future_done

            states.append(state)
            actions.append(action)
            rewards.append(total_reward)
            next_states.append(current_next_state)
            dones.append(current_done)

        # Convert lists of tuples to NumPy arrays with the correct shape
        states = np.array(states, dtype=np.float32)  # states are shaped as (batch_size, state_dim)
        next_states = np.array(next_states, dtype=np.float32)

        # Conversion to tensors
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions).view(-1, 1)  # Ensure actions are properly shaped for gather
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # Compute Q values for the current states and actions
        current_q_values = self.model(states).gather(1, actions).squeeze(1)

        # Compute the maximum Q values for the next states
        next_q_values = self.target_model(next_states).max(1)[0]

        # Compute the expected Q values
        expected_q_values = rewards + (1 - dones) * self.gamma ** n_steps * next_q_values

        # Compute loss
        loss = self.loss_fn(current_q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


# after all the training finished
def infer_best_route(agent, optimizer, env, max_steps=1000):
    state = env.reset()
    best_route = [env.current_node]
    best_modes = []
    total_time_cost = 0
    steps = 0
    find = False

    while steps < max_steps:
        current_node = env.current_node
        possible_actions = [(neighbor, key, data) for neighbor in env.graph.neighbors(env.current_node)
                            for key, data in env.graph[env.current_node][neighbor].items()]
        # print(possible_actions)
        num_available_actions = len(possible_actions)
        if num_available_actions == 0:
            break

        action = agent.act(state, num_available_actions, test=True)

        next_state, reward, done, info = env.step(action)
        if info['action_taken'] == 'Loop detected':
            return best_route, best_modes, total_time_cost, find
        distance = optimizer.new_graph[current_node][env.current_node][info['mode']]['distance']
        time_cost = optimizer.new_graph[current_node][env.current_node][info['mode']]['weight']
        print(distance, time_cost)

        if done:
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


def run_dqn(optimizer, source_edge, target_edge, episode_number):
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

    env = Environment(optimizer.new_graph, source_edge, target_edge)
    agent = DQNAgent(state_dim, action_dim)
    start_time = time.time()
    update_frequency = 10

    for episode in range(episode_number):
        total_reward = 0
        state = env.reset()
        route = [env.current_node]
        done = False
        total_rewards = 0
        rewards_count = []
        modes = []

        while not done:
            if len(route) >= 2 and route[1] == '23395388#5':
                print("hehe")
            possible_actions = [
                (neighbor, key, data)
                for neighbor in env.graph.neighbors(env.current_node)
                for key, data in env.graph[env.current_node][neighbor].items()
            ]
            num_available_actions = len(possible_actions)
            action = agent.act(state, num_available_actions, test=False)
            _, mode, _ = possible_actions[action]
            next_state, reward, done, info = env.step(action)
            route.append(env.current_node)
            modes.append(mode)
            rewards_count.append(reward)
            total_rewards -= reward
            agent.remember(state, action, reward, next_state, done)

            if next_state == state:
                break  # Avoid looping in the same state
            state = next_state
            total_reward += reward
            #
            # if done:
            #     print("=====================================")
            #     print(modes)
            #     print(route)
            #     print(total_rewards)
            #     print("=====================================")

            if len(agent.memory) > 32:
                agent.replay()

        if (episode + 1) % update_frequency == 0:
            agent.update_target_model()

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

    return best_route, best_modes, total_time_cost, execution_time, find
