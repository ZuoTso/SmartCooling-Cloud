import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# Load the environment
import air_conditioning_env as ac_env

# ------------------------------
# Defining the DQN network model
# ------------------------------
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# ------------------------------
# Prioritized Experience Replay Buffer
# ------------------------------
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-6):
        self.capacity = int(capacity)
        self.buffer = []
        self.pos = 0
        # Stores the priority of each transition
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.alpha = alpha  # Controls how the priorities are distributed (0 equals uniform sampling)
        self.beta = beta    # Used for importance sampling weights, which are gradually increased with training
        self.beta_increment = beta_increment
        self.epsilon = epsilon  # Prevent priority from being 0

    def add(self, transition):
        # When the current buffer is not empty, use the current maximum priority, otherwise initialize to 1.0
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.buffer)]
        # Calculate the sampling probability of each transition
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        # Update beta, gradually approaching 1
        self.beta = np.min([1.0, self.beta + self.beta_increment])
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon

# ------------------------------
# Defining the DQN Agent with Target Network and Prioritized Experience Replay
# ------------------------------
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer(capacity=1e4)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.95
        self.learning_rate = 1e-6
        self.update_target_every = 1000  # Target Network is updated every 1000 steps
        self.step_count = 0

        # Make sure the GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build the main network & target network
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())  # Synchronize weights at initialization
        self.target_model.eval()  # The target network does not require backpropagation

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.loss_fn = nn.MSELoss(reduction='none')  # Use no averaging for subsequent weighting

    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        if len(self.memory.buffer) < batch_size:
            return

        # Sampling from PrioritizedReplayBuffer
        minibatch, indices, is_weights = self.memory.sample(batch_size)
        # Split the data in the batch
        states = np.vstack([transition[0] for transition in minibatch])
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        next_states = np.vstack([transition[3] for transition in minibatch])
        dones = np.array([transition[4] for transition in minibatch]).astype(np.float32)

        states_tensor = torch.FloatTensor(states).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        is_weights_tensor = torch.FloatTensor(is_weights).to(self.device)

        # Current Q value: The Q value of the action taken from the main network
        current_q = self.model(states_tensor).gather(1, actions_tensor).squeeze(1)
        # Calculate the target Q value: Get the maximum Q value of the next state from the target network
        with torch.no_grad():
            next_q = self.target_model(next_states_tensor).max(1)[0]
        target_q = rewards_tensor + self.gamma * next_q * (1 - dones_tensor)

        # Calculate TD error and weight the loss using importance sampling weights
        td_errors = current_q - target_q
        loss = (is_weights_tensor * self.loss_fn(current_q, target_q)).mean()

        self.optimizer.zero_grad()
        with torch.autograd.detect_anomaly():
            loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update each transition priority (based on |TD error|)
        new_priorities = np.abs(td_errors.detach().cpu().numpy())
        self.memory.update_priorities(indices, new_priorities)

        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()
