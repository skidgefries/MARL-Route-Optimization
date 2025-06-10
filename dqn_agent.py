import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# --- Q-Network ---
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return self.fc3(x)

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, buffer_size=10000):
        self.memory = deque(maxlen=buffer_size)

    def add(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99, buffer_size=10000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size

        # Networks
        self.qnetwork = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.update_target_network()

        # Optimizer + loss
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Exploration params
        self.epsilon = 1.0  # Start fully random
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995

    def act(self, state, use_epsilon=True):
        state = torch.FloatTensor(state).unsqueeze(0)
        self.qnetwork.eval()
        with torch.no_grad():
            q_values = self.qnetwork(state)
        self.qnetwork.train()

        if use_epsilon and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            return torch.argmax(q_values).item()


    def step(self, state, action, reward, next_state, done):
        """Store experience + Learn"""
        self.replay_buffer.add((state, action, reward, next_state, done))
        if len(self.replay_buffer) >= self.batch_size:
            self.learn()

    def learn(self):
        """Sample from buffer and update network"""
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)


        # Current Q estimates
        q_values = self.qnetwork(states).gather(1, actions)

        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + self.gamma * next_q_values * (1 - dones)

        # Loss
        loss = self.loss_fn(q_values, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """Copy weights to target network"""
        self.target_network.load_state_dict(self.qnetwork.state_dict())

    def decay_epsilon(self):
        """Reduce exploration over time"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
