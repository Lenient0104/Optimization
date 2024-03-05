import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
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
        self.reset()

    def reset(self):
        self.current_node = self.source
        self.remaining_energy = self.initial_energy
        return self._get_state()

    def _get_state(self):
        # State representation could be more complex depending on your exact scenario
        # Here, it's simplified as a vector containing the current node and remaining energy
        current_node_index = list(self.graph.nodes).index(self.current_node)
        state = np.array([current_node_index, self.remaining_energy])
        return state

    def step(self, action):
        # Assuming action is an index for a selected edge from the current node
        possible_actions = list(self.graph.out_edges(self.current_node, data=True))
        selected_edge = possible_actions[action]

        # Unpack selected action details
        _, next_node, data = selected_edge
        mode = data['mode']
        distance = data['distance']  # Assuming distance or weight information is stored in edge data

        # Calculate energy consumption
        energy_consumed = self.calculate_energy_comsumption(mode, distance)

        # Update environment state
        reward = -1  # Simple reward, customize based on your requirements
        done = False
        if next_node == self.destination:
            done = True
            reward = 100  # Reward for reaching the destination
        elif self.remaining_energy - energy_consumed <= 0:
            done = True
            reward = -100  # Penalty for running out of energy

        self.remaining_energy -= energy_consumed
        self.current_node = next_node

        next_state = self._get_state()
        return next_state, reward, done

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
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
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
source_edge = '3789374#3'
destination_edge = "-361450282"
user = User(60, False, 0, 20)
optimizer = Optimization(net_xml_path, user, 'test_new.db', source_edge, destination_edge)
graph = optimizer.new_graph
env = Environment(graph, source_edge, destination_edge)

# Define state and action dimensions
state_dim = 2  # For example: [current_node_index, remaining_energy]
action_dim = max(
    len(list(graph.out_edges(node))) for node in graph.nodes)  # Max number of possible actions from any node

agent = DQNAgent(state_dim, action_dim)

# Training loop
for episode in range(100):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)

        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if done:
            agent.update_target_model()

    if len(agent.memory) > 32:
        agent.replay()

    print(f"Episode {episode + 1}: Total Reward: {total_reward}")
