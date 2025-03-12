import matplotlib.pyplot as plt
import numpy as np
import air_conditioning_env as ac_env
import statistics

# 若想載入模型：
csv_path = "test data_2_11.csv"
env = ac_env.AirConditioningEnv(csv_path)

# Get the dimensions of the environment state and action
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

# 1. 建立模型架構（與訓練時的架構相同）
model = DQN(state_size, action_size).to(agent.device)
# 2. 載入權重
agent.model.load_state_dict(torch.load('dqn_model_v5.pth'))
agent.model.eval()  # 設定為 evaluation 模式


test_data_path = "test data_2_11.csv"
hours = 336
seed = 2
np.random.seed(seed)
torch.manual_seed(seed)
agent.epsilon = 0  # 確保不探索

# 建立兩個環境實例，分別用於 DQN agent 與 Heuristic 策略
env_dqn = ac_env.AirConditioningEnv(test_data_path, episode_length=hours)
env_heuristic = ac_env.AirConditioningEnv(test_data_path, episode_length=hours)

state_size = env_dqn.observation_space.shape[0]
action_size = env_dqn.action_space.n

# 載入或訓練好的 DQN agent（這裡假設 agent 已訓練完成）
# agent = DQNAgent(state_size, action_size)
# 若有保存模型權重，可在此載入

def heuristic_policy(state):
    """
    基於政府公家機關的策略：
    - 當室外溫度 > 28℃，期望室內設定為 27℃：
         若目前室內溫度 > 27，採取 action 0 (降低)；
         若 < 27，採取 action 2 (提高)；
         等於 27 則維持 (action 1)。
    - 當室外溫度 < 27℃，則關閉冷氣（目標設定溫度達到 30℃）：
         若目前室內溫度 < 30，採取 action 2 (提高)；
         否則維持 (action 1)。
    - 其他情況則維持 (action 1)。
    """
    T_in = state[0]    # 室內冷氣設定溫度
    T_out = state[1]   # 室外溫度
    T_ac = 27       # 26~28℃
    if T_out > 28:
        if T_in > T_ac:
            return 0  # 降低空調設定溫度
        elif T_in < T_ac:
            return 2  # 提高空調設定溫度
        else:
            return 1  # 維持
    elif T_out < T_ac:
        if T_in < 30:
            return 2  # 提高至關閉冷氣（室內溫度達到 30℃）
        else:
            return 1  # 維持
    else:
        return 1  # 其他情況維持

def run_episode(env, policy_func, max_steps=hours):
    """
    根據給定的 policy (可為 heuristic_policy 或 agent.act)
    執行單一回合並記錄每個 step 的資訊：
    - 累計耗電量（從 info["energy"] 取得）
    - THI（從 info["THI"] 取得）
    - 室內溫度（state[4]）
    - 室外溫度（state[1]）
    """
    # 使用相同 seed 重置環境，確保起始點一致
    state = env.reset(seed=seed)
    state = np.reshape(state, [1, env.observation_space.shape[0]])
    steps = []
    energies = []      # 累計耗電量
    THI_values = []    # 舒適度 THI
    T_in_values = []   # 室內溫度
    T_out_values = []  # 室外溫度

    for step in range(max_steps):
        # 如果 policy_func 是 agent.act，傳入 2D state；否則傳入 1D state
        if policy_func == agent.act:
            action = policy_func(state)
        else:
            action = policy_func(state[0])
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
        state = next_state

        steps.append(step)
        energies.append(info.get("energy", 0))
        THI_values.append(info.get("THI", 0))
        T_in_values.append(state[0, 4])
        T_out_values.append(state[0, 1])
        if done:
            break
    return steps, energies, THI_values, T_in_values, T_out_values

# 分別用 DQN agent 與 Heuristic 策略執行回合，兩個環境皆以 seed 重置，確保天氣資料相同
dqn_steps, dqn_energies, dqn_THI, dqn_Tin, dqn_Tout = run_episode(env_dqn, agent.act)
heuristic_steps, heuristic_energies, heuristic_THI, heuristic_Tin, heuristic_Tout = run_episode(env_heuristic, heuristic_policy)

# 繪圖比較 (2x2 子圖)
plt.figure(figsize=(14, 10))

# 1. 累計耗電量曲線
plt.subplot(2, 2, 1)
plt.plot(dqn_steps, dqn_energies, label="DQN Agent")
plt.plot(heuristic_steps, heuristic_energies, label="Heuristic", linestyle="--")
plt.xlabel("Steps")
plt.ylabel("Cumulative Energy (Wh)")
plt.title("Energy Consumption Curve")
plt.legend()

# 2. THI 舒適度曲線
plt.subplot(2, 2, 2)
plt.plot(dqn_steps, dqn_THI, label="DQN Agent")
plt.plot(heuristic_steps, heuristic_THI, label="Heuristic", linestyle="--", color="orange")
plt.xlabel("Steps")
plt.ylabel("THI")
plt.title("THI Curve")
plt.legend()
print(f"Mean DQN |THI - 23|:{abs(statistics.mean(dqn_THI) - 23):>14.4f}")
print(f"Mean Heuristic |THI - 23|:{abs(statistics.mean(heuristic_THI) - 23):>8.4f}")

# 3. 冷氣設定溫度 (T_ac) 曲線
plt.subplot(2, 2, 3)
plt.plot(dqn_steps, dqn_Tin, label="DQN Agent")
plt.plot(heuristic_steps, heuristic_Tin, label="Heuristic", linestyle="--", color="orange")
plt.xlabel("Steps")
plt.ylabel("Indoor Temperature (°C)")
plt.title("Indoor Temperature Curve")
plt.legend()

# 4. 室外溫度 (T_out) 曲線
plt.subplot(2, 2, 4)
plt.plot(dqn_steps, dqn_Tout, label="DQN Agent")
plt.plot(heuristic_steps, heuristic_Tout, label="Heuristic", linestyle="--", color="orange")
plt.xlabel("Steps")
plt.ylabel("Outdoor Temperature (°C)")
plt.title("Outdoor Temperature Curve")
plt.legend()

plt.tight_layout()
plt.show()
