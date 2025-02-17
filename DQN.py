import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# 載入你定義的環境，請確保路徑正確
import air_conditioning_env as ac_env

# ------------------------------
# 定義 DQN 網路模型
# ------------------------------
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.out = nn.Linear(24, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# ------------------------------
# 定義 DQN Agent
# ------------------------------
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size      # 狀態維度，例如 3
        self.action_size = action_size    # 動作數量，例如 3
        self.memory = deque(maxlen=2000)    # replay memory
        self.gamma = 0.95                 # 折扣因子
        self.epsilon = 1.0                # 初始探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        # 判斷是否有 GPU 可用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        # state shape: (1, state_size)，為 numpy 陣列
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
            
            # 初始 target 為 reward，若非終止狀態則加上未來折扣
            target = reward
            if not done:
                with torch.no_grad():
                    next_q = self.model(next_state_tensor)
                    target = reward + self.gamma * torch.max(next_q).item()
                    
            # 取得當前預測的 Q 值
            q_val = self.model(state_tensor)
            target_f = q_val.clone().detach()
            # 修改這裡，正確更新第 0 個 batch 裡面的對應動作
            target_f[0, action] = target
            
            states.append(state_tensor)
            targets.append(target_f)
        
        # 將 states 與 targets 合併成 batch
        states = torch.cat(states, dim=0)
        targets = torch.cat(targets, dim=0)
        
        self.optimizer.zero_grad()
        predictions = self.model(states)
        loss = self.loss_fn(predictions, targets)
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ------------------------------
# 主程式：訓練迴圈與結果可視化
# ------------------------------
import matplotlib.pyplot as plt
if __name__ == "__main__":
    # 請替換為你的 CSV 檔案路徑，並確保 CSV 包含「氣溫(℃)」與「AH(g/m³)」欄位
    csv_path = "training data.csv"  
    env = ac_env.AirConditioningEnv(csv_path)
    
    # 取得環境狀態與動作的維度
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    
    episodes = 500  # 訓練總回合數
    batch_size = 32

    # 訓練迴圈
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(3000):  # 每回合最多 n 步
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            if done:
                print(f"Episode: {e+1}/{episodes}, Steps: {time+1}, Epsilon: {agent.epsilon:.2f}")
                break
            if len(agent.memory) >= batch_size:
                agent.replay(batch_size)

# 定義單一回合測試並記錄數據 (用於可視化)
def run_episode(env, agent, max_steps=500):
    state = env.reset()  # reset() 必須回傳初始 state
    state = np.reshape(state, [1, env.observation_space.shape[0]])
    
    steps = []
    energies = []    # 累計耗電量
    THI_values = []  # 舒適度(THI值)
    
    for step in range(max_steps):
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
        
        # 將每個步驟的經驗存入記憶(可選，視是否需要進行額外訓練)
        agent.remember(state, action, reward, next_state, done)
        
        # 從 next_state 中取得累計耗電量(假設其 index 為 2)
        cumulative_energy = next_state[0, 2]
        steps.append(step)
        energies.append(cumulative_energy)
        # info 字典中我們會傳回 THI(需在環境 step() 中加入 info 回傳)
        THI_values.append(info.get("THI", 0))
        
        state = next_state
        if done:
            break
    return steps, energies, THI_values

# 訓練結束後，執行一個回合並記錄數據以便可視化
steps, energies, THI_values = run_episode(env, agent, max_steps=500)

# 使用 matplotlib 畫圖
plt.figure(figsize=(15, 5))

# 繪製累計耗電量曲線
plt.subplot(1, 2, 1)
plt.plot(steps, energies, label='Cumulative power consumption (Wh)')
plt.xlabel('steps')
plt.ylabel('Cumulative power consumption (Wh)')
plt.title('power consumption curve')
plt.legend()

# 繪製舒適度 (THI) 曲線
plt.subplot(1, 2, 2)
plt.plot(steps, THI_values, label='THI', color='orange')
plt.xlabel('steps')
plt.ylabel('THI')
plt.title('THI curve')
plt.legend()

plt.tight_layout()
plt.show()