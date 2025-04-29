import matplotlib.pyplot as plt
import numpy as np
import air_conditioning_env as ac_env
import statistics

test_data_path = "test data_2_11.csv"
hours = 336
seed = 999
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
         若目前室內溫度 > 27，採取 action 2,3 (降低)；
         若 < 27，採取 action 4,5 (提高)；
         等於 27 則維持 (action 1)。
    - 當室外溫度 < 27℃，則關閉冷氣：
    - 其他情況則維持 (action 1)。
    """
    T_in = state[4]    # 室內溫度
    T_out = state[1]   # 室外溫度
    T_ac = 27       # 26~28℃
    if T_out > 28:
        if T_in > T_ac:
            return 2  # 降低空調設定溫度
        elif T_in < T_ac:
            return 4  # 提高空調設定溫度
        else:
            return 1  # 維持
    elif T_out < T_ac:
        return 0  # ac turn off
    else:
        return 0  # ac turn off

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
    dates = []
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
        dates.append(info.get("date", 0))
        T_in_values.append(state[0, 4])
        T_out_values.append(state[0, 1])
        if done:
            break
    return steps, energies, THI_values, T_in_values, T_out_values, dates

# 分別用 DQN agent 與 Heuristic 策略執行回合，兩個環境皆以 seed 重置，確保天氣資料相同
dqn_steps, dqn_energies, dqn_THI, dqn_Tin, dqn_Tout, dqn_dates = run_episode(env_dqn, agent.act)
heuristic_steps, heuristic_energies, heuristic_THI, heuristic_Tin, heuristic_Tout, heuristic_dates = run_episode(env_heuristic, heuristic_policy)

# 繪圖比較 (2x2 子圖)
plt.figure(figsize=(14, 10))

# 1. 累計耗電量曲線
plt.subplot(2, 2, 1)
plt.plot(dqn_steps, dqn_energies, label="DQN Agent")
plt.plot(heuristic_steps, heuristic_energies, label="Heuristic", linestyle="--")
plt.xlabel("Steps")
plt.ylabel("Cumulative Energy (kWh)")
plt.title("Energy Consumption Curve")
plt.legend()

# 2. THI 舒適度曲線
plt.subplot(2, 2, 2)

ax = plt.gca()
ax.set_ylim(14, 28)

ax.axhspan(22, 24, facecolor='green', alpha=0.3)
# ax.axhspan(19, 22, facecolor='yellow', alpha=0.3)
# ax.axhspan(24, 27, facecolor='yellow', alpha=0.3)
# ax.axhspan(15, 19, facecolor='orange', alpha=0.3)
# ax.axhspan(10, 15, facecolor='red', alpha=0.3)
# ax.axhspan(27, 30, facecolor='red', alpha=0.3)

plt.plot(dqn_steps, dqn_THI, label="DQN Agent")
plt.plot(heuristic_steps, heuristic_THI, label="Heuristic", linestyle="--", color="orange")
plt.xlabel("Steps")
plt.ylabel("THI")
plt.title("THI Curve")
plt.legend()

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

heuristic_count_22_24 = sum(1 for val in heuristic_THI if 22 <= val <= 24)
heuristic_count_19_27 = sum(1 for val in heuristic_THI if 19 <= val <= 27)

heuristic_prop_22_24 = heuristic_count_22_24 / len(heuristic_THI)
heuristic_prop_19_27 = heuristic_count_19_27 / len(heuristic_THI)

dqn_count_22_24 = sum(1 for val in dqn_THI if 22 <= val <= 24)
dqn_count_19_27 = sum(1 for val in dqn_THI if 19 <= val <= 27)

dqn_prop_22_24 = dqn_count_22_24 / len(dqn_THI)
dqn_prop_19_27 = dqn_count_19_27 / len(dqn_THI)

print(f"start date: {dqn_dates[0]}")
print(f"{'heuristic ':.>10}總耗電量：{heuristic_energies[-1]:>5.3f} kWh")
print(f"{'dqn ':>10}總耗電量：{dqn_energies[-1]:>5.3f} kWh")
print("--------------------------")
print(f"{'heuristic ':.>10}最佳舒適占比(22~24)：{heuristic_prop_22_24:.2f}")
print(f"{'dqn ':>10}最佳舒適占比(22~24)：{dqn_prop_22_24:.2f}")
print("--------------------------")
print(f"{'heuristic ':.>10}舒適占比(19~27)：{heuristic_prop_19_27:.2f}")
print(f"{'dqn ':>10}舒適占比(19~27)：{dqn_prop_19_27:.2f}")