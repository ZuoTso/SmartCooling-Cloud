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
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        # 儲存每個 transition 的優先權
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha  # 控制優先權的分布程度（0 等同均勻采樣）
        self.beta = beta    # 用於 importance sampling 權重，隨訓練逐步提升
        self.beta_increment = beta_increment
        self.epsilon = epsilon  # 防止優先權為 0

    def add(self, transition):
        # 當前 buffer 不為空時，使用目前最大優先級，否則初始化為 1.0
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
        # 計算各筆 transition 的抽樣機率
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        # 更新 beta，逐步趨近 1
        self.beta = np.min([1.0, self.beta + self.beta_increment])
        # 計算 importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # 歸一化
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
        self.memory = PrioritizedReplayBuffer(capacity=7000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.95
        self.learning_rate = 1e-6
        self.update_target_every = 1000  # Target Network 每 1000 步更新一次
        self.step_count = 0

        # 確保 GPU 可用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 建立主網絡 & 目標網絡
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())  # 初始化時同步權重
        self.target_model.eval()  # 目標網絡不需要反向傳播

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.loss_fn = nn.MSELoss(reduction='none')  # 使用不取平均以便後續加權

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

        # 從 PrioritizedReplayBuffer 中抽樣
        minibatch, indices, is_weights = self.memory.sample(batch_size)
        # 將 batch 中的資料拆解
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

        # 當前 Q 值：從主網絡取出採取動作的 Q 值
        current_q = self.model(states_tensor).gather(1, actions_tensor).squeeze(1)
        # 計算 target Q 值：從 target 網絡獲取下一狀態的最大 Q 值
        with torch.no_grad():
            next_q = self.target_model(next_states_tensor).max(1)[0]
        target_q = rewards_tensor + self.gamma * next_q * (1 - dones_tensor)

        # 計算 TD 誤差，並利用 importance sampling 權重加權損失
        td_errors = current_q - target_q
        loss = (is_weights_tensor * self.loss_fn(current_q, target_q)).mean()

        self.optimizer.zero_grad()
        with torch.autograd.detect_anomaly():
            loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 更新每筆 transition 的優先級 (以 |TD error| 作為依據)
        new_priorities = np.abs(td_errors.detach().cpu().numpy())
        self.memory.update_priorities(indices, new_priorities)

        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

# ------------------------------
# Main: training loop
# ------------------------------
import matplotlib.pyplot as plt
if __name__ == "__main__":

    csv_path = "training data.csv"
    episode_length = 504 # Set one episode per day(24), week(168), two weeks(336), three weeks(504), four weeks(672) as needed
    env = ac_env.AirConditioningEnv(csv_path, episode_length=episode_length)
    env = NormalizeObservation(env)

    # Get the dimensions of the environment state and action
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    episodes = 200  # Total number of training rounds
    batch_size = 32
    losses_per_episode = []  # Used to record the average loss per round
    cumulative_rewards = []     # Record cumulative reward per episode
    avg_THI_deviation_per_episode = []
    avg_power_consumption_per_episode = []
    peak_power_consumption_per_episode = []

    # Training Cycles
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        episode_losses = []  # The loss of all replays in this round
        cumulative_reward = 0   # Initialize cumulative reward for the episode
        THI_deviations = []   # |THI - 23|
        instantaneous_powers = []
        prev_energy = 0

        for time in range(episode_length):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            cumulative_reward += reward

            # |THI - 23|
            current_THI = info["THI"]
            THI_deviation = abs(current_THI - 23)
            THI_deviations.append(THI_deviation)

            # Calculate instantaneous energy consumption
            current_energy = info["energy"]
            instantaneous_power = current_energy - prev_energy
            instantaneous_powers.append(instantaneous_power)
            prev_energy = current_energy

            if np.isnan(reward):
                print("NaN detected in reward!")
                print(f"reward: {reward}")
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                break
            loss = agent.replay(batch_size)
            if loss is not None:
                episode_losses.append(loss)

        # Calculate the average THI deviation for this round
        avg_THI_deviation = np.mean(THI_deviations) if THI_deviations else 0
        avg_THI_deviation_per_episode.append(avg_THI_deviation)

        # Calculate the average instantaneous energy consumption and peak energy consumption of this round
        avg_power_consumption = np.mean(instantaneous_powers) if instantaneous_powers else 0
        peak_power_consumption = np.max(instantaneous_powers) if instantaneous_powers else 0
        avg_power_consumption_per_episode.append(avg_power_consumption)
        peak_power_consumption_per_episode.append(peak_power_consumption)
        
        # Calculate the average loss of the current round
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        losses_per_episode.append(avg_loss)
        cumulative_rewards.append(cumulative_reward)
        print(f"Episode:{e+1:>4}/{episodes:<4}, Avg Loss: {avg_loss:.4f}, Cumulative Reward: {cumulative_reward:>10.4f}, "
          f"Avg |THI-23|: {avg_THI_deviation:.4f}, Avg Power: {avg_power_consumption:.4f}, Peak Power: {peak_power_consumption:.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(range(1, episodes+1), losses_per_episode, label='Average Loss per Episode')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(1, episodes+1), avg_THI_deviation_per_episode, label='Average |THI-23| per Episode')
plt.xlabel('Episode')
plt.ylabel('Average |THI-23|')
plt.title('Average |THI-23| Curve')
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

plt.figure(figsize=(10, 5))
plt.plot(range(1, episodes+1), avg_power_consumption_per_episode, label='Average Power Consumption per Episode')
plt.xlabel('Episode')
plt.ylabel('Average Power Consumption')
plt.title('Average Power Consumption Curve')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(1, episodes+1), peak_power_consumption_per_episode, label='Peak Power Consumption per Episode')
plt.xlabel('Episode')
plt.ylabel('Peak Power Consumption')
plt.title('Peak Power Consumption Curve')
plt.legend()
plt.show()