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
# Defining the DQN Agent
# ------------------------------
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size      # Status Dimension
        self.action_size = action_size    # Action Dimension
        self.memory = deque(maxlen=2000)    # replay memory
        self.gamma = 0.95                 # Discount Factor
        self.epsilon = 1.0                # Initial exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 1e-4

        # Determine whether a GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # state shape: (1, state_size)，is numpy array
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states = []
        targets = []

        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).to(self.device)

            # The initial target is the reward, and if it is not a terminal state, the future discount is added
            target = reward
            if not done:
                with torch.no_grad():
                    next_q = self.model(next_state_tensor)
                    target = reward + self.gamma * torch.max(next_q).item()

            # Get the current predicted Q value
            q_val = self.model(state_tensor)
            target_f = q_val.clone().detach()

            target_f[0, action] = target

            # 除錯訊息：檢查 state, reward, q_val, target_f 是否有 NaN
            if torch.isnan(state_tensor).any():
                print("NaN detected in state_tensor!")
                print(f"state_tensor: {state_tensor}")
            # if np.isnan(reward):
            #     print("NaN detected in reward!")
            #     print(f"reward: {reward}")
            if torch.isnan(q_val).any():
                print("NaN detected in q_val!")
                print(f"q_val: {q_val}")
            if torch.isnan(target_f).any():
                print("NaN detected in target_f!")
                print(f"target_f: {target_f}")

            states.append(state_tensor)
            targets.append(target_f)

        # Combine states and targets into batches
        states = torch.cat(states, dim=0)
        targets = torch.cat(targets, dim=0)

        self.optimizer.zero_grad()
        predictions = self.model(states)
        loss = self.loss_fn(predictions, targets)

        with torch.autograd.detect_anomaly():
          loss.backward()


        # # 觀察各參數的梯度統計資訊
        # for name, param in self.model.named_parameters():
        #     if param.grad is not None:
        #         grad_mean = param.grad.mean().item()
        #         grad_max = param.grad.max().item()
        #         grad_min = param.grad.min().item()
        #         # print(f"{name} grad - mean: {grad_mean:.6f}, max: {grad_max:.6f}, min: {grad_min:.6f}")
        #         # 檢查是否出現 NaN
        #         if torch.isnan(param.grad).any():
        #             print(f"NaN detected in gradients of {name}!")

        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss.item()  # Return the loss value of the current training

# ------------------------------
# Main: training loop
# ------------------------------
import matplotlib.pyplot as plt
if __name__ == "__main__":

    csv_path = "training data.csv"
    env = ac_env.AirConditioningEnv(csv_path)
    env = NormalizeObservation(env)

    # Get the dimensions of the environment state and action
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    episodes = 250  # Total number of training rounds
    batch_size = 32
    losses_per_episode = []  # Used to record the average loss per round
    cumulative_rewards = []     # Record cumulative reward per episode

    # Training Cycles
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        episode_losses = []  # The loss of all replays in this round
        cumulative_reward = 0   # Initialize cumulative reward for the episode

        for time in range(200):  # Each round has at most n steps
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            cumulative_reward += reward
            if np.isnan(reward):
                print("NaN detected in reward!")
                print(f"reward: {reward}")
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print(f"Episode: {e+1}/{episodes}, Steps: {time+1}, Epsilon: {agent.epsilon:.2f}")
                break
            if len(agent.memory) >= batch_size:
                loss = agent.replay(batch_size)
                if loss is not None:
                    episode_losses.append(loss)

        # Calculate the average loss of the current round
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        losses_per_episode.append(avg_loss)
        cumulative_rewards.append(cumulative_reward)
        print(f"Episode:{e+1:>4}/{episodes:<4}, Average Loss: {avg_loss:.4f}, Cumulative Reward:{cumulative_reward:>10.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(range(1, episodes+1), losses_per_episode, label='Average Loss per Episode')
# plt.plot(range(1, 31), losses_per_episode, label='Average Loss per Episode')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()

# Plot Cumulative Reward Curve
plt.figure(figsize=(10, 5))
plt.plot(range(1, episodes+1), cumulative_rewards, label='Cumulative Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward Curve')
plt.legend()
plt.show()