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
# Defining the DQN Agent with Target Network
# ------------------------------
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=7000)
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
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
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

            # 計算目標值 (使用 target_model)
            target = reward
            if not done:
                with torch.no_grad():
                    next_q = self.target_model(next_state_tensor)  # 使用 Target Network
                    target = reward + self.gamma * torch.max(next_q).item()

            # 更新 Q 值
            q_val = self.model(state_tensor)
            target_f = q_val.clone().detach()
            target_f[0, action] = target

            states.append(state_tensor)
            targets.append(target_f)

        # 組成 batch 更新
        states = torch.cat(states, dim=0)
        targets = torch.cat(targets, dim=0)

        self.optimizer.zero_grad()
        predictions = self.model(states)
        loss = self.loss_fn(predictions, targets)
        with torch.autograd.detect_anomaly():
          loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

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

        # 每 `update_target_every` 步更新 Target Network
        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        # 逐步降低 ε
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()  # 回傳 loss

# ------------------------------
# Main: training loop
# ------------------------------
import matplotlib.pyplot as plt
if __name__ == "__main__":

    csv_path = "training data.csv"
    episode_length = 672 # Set one episode per day(24), week(168), two weeks(336), three weeks(504), four weeks(672) as needed
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
            next_state, reward, done, _ = env.step(action)
            cumulative_reward += reward

            # |THI - 23|
            current_THI = _["THI"]
            THI_deviation = abs(current_THI - 23)
            THI_deviations.append(THI_deviation)

            # Calculate instantaneous energy consumption
            current_energy = _["energy"]
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
            if len(agent.memory) >= batch_size:
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

# Plot Cumulative Reward Curve
plt.figure(figsize=(10, 5))
plt.plot(range(1, episodes+1), cumulative_rewards, label='Cumulative Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward Curve')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(1, episodes+1), avg_THI_deviation_per_episode, label='Average |THI-23| per Episode')
plt.xlabel('Episode')
plt.ylabel('Average |THI-23|')
plt.title('Average |THI-23| Curve')
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