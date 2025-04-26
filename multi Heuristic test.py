import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats

# 固定所有可能的隨機性來源
def set_all_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 在程式開始時固定一個全局 seed
GLOBAL_SEED = 42
set_all_seeds(GLOBAL_SEED)

# 載入自訂環境與 DQN Agent 模組
import air_conditioning_env as ac_env
from DQN import DQNAgent

# ---------------------------
# 設定參數
# ---------------------------
test_model = 'dqn_model_v6.5.pth'
test_data_path = "test data_2_11.csv"  # 測試資料 CSV
episode_length = 336  # 一回合 336 小時（14 天）
num_trials = 20       # 試驗次數
seed_base = 200       # 起始 seed 值，每個 trial 傳入不同的 seed，但每次跑同一個 trial 都應固定

# ---------------------------
# 建立環境及 DQN Agent
# ---------------------------
# 利用 dummy 環境取得狀態與動作空間維度
env_dummy = ac_env.AirConditioningEnv(test_data_path, episode_length=episode_length)
state_size = env_dummy.observation_space.shape[0]
action_size = env_dummy.action_space.n

# # 假設 DQNAgent 已訓練完成，這裡僅用於評估，不進行探索
# agent = DQNAgent(state_size, action_size)
# agent.epsilon = 0  # 關閉隨機探索

# 1. 建立模型架構（與訓練時的架構相同）
model = DQN(state_size, action_size).to(agent.device)
# 2. 載入權重
agent.model.load_state_dict(torch.load(test_model, map_location=agent.device))
agent.model.eval()  # 設定為 evaluation 模式

# ---------------------------
# 定義 Heuristic 策略
# ---------------------------
def heuristic_policy(state):
    """
    基於政府公家機關的策略：
      - 當室外溫度 > 28℃ 時，期望室內設定為 27℃：
            如果目前室內溫度 > 27，採取 action 0（降低設定溫度）；
            如果低於 27，採取 action 2（提高設定溫度）；
            等於 27 則採取 action 1（維持）。
      - 當室外溫度 < 27℃ 時，則關閉冷氣（目標室內溫度為 30℃）：
            如果目前室內溫度 < 30，採取 action 2（提高設定溫度）；
            否則採取 action 1。
      - 其他情況則採取 action 1 (維持)。
    """
    T_in = state[0]    # 室內冷氣設定溫度
    T_out = state[1]   # 室外溫度
    target = 27
    if T_out > 28:
        if T_in > target:
            return 0
        elif T_in < target:
            return 2
        else:
            return 1
    elif T_out < target:
        if T_in < 30:
            return 2
        else:
            return 1
    else:
        return 1

# ---------------------------
# 定義單回合評估函數
# ---------------------------
def run_episode(env, policy_func, agent=None, max_steps=episode_length, seed=None):
    """
    執行單一回合，並回傳：
      - 累計耗電量（從 info["energy"] 取得）
      - 該回合所有 time step 的 THI 值（list 回傳）
    若使用 DQN 策略，傳入 2D 狀態；若使用 heuristic 策略，則傳入 1D 狀態。
    在 reset 時傳入 seed，環境內部也會用 np.random.seed(seed) 固定起始狀態。
    """
    # 固定本次試驗的 seed（全局也可在此處再次設定）
    set_all_seeds(seed)
    state = env.reset(seed=seed)
    state = np.reshape(state, [1, env.observation_space.shape[0]])
    cumulative_energy = 0
    thi_list = []
    for step in range(max_steps):
        if agent is not None and policy_func == agent.act:
            action = agent.act(state)
        else:
            action = policy_func(state[0])
        next_state, reward, done, info = env.step(action)
        cumulative_energy = info.get("energy", cumulative_energy)
        thi = info.get("THI", 0)
        thi_list.append(thi)
        state = np.reshape(next_state, [1, env.observation_space.shape[0]])
        if done:
            break
    return cumulative_energy, thi_list

# ---------------------------
# 多次試驗與數據收集
# ---------------------------
# 儲存累計耗電量與 THI 占比
dqn_energies = []
heuristic_energies = []

# 儲存 THI 在 22~24 及 19~27 區間的比例
dqn_prop_22_24_list = []
dqn_prop_19_27_list = []
heuristic_prop_22_24_list = []
heuristic_prop_19_27_list = []

for trial in range(num_trials):
    trial_seed = seed_base + trial  # 每次 trial 的 seed 會固定，只要程式重新跑，結果應一致
    # 建立兩個環境實例，並以相同 seed 重置，確保 DQN 與 Heuristic 在相同條件下評估
    env_dqn = ac_env.AirConditioningEnv(test_data_path, episode_length=episode_length)
    env_heuristic = ac_env.AirConditioningEnv(test_data_path, episode_length=episode_length)

    energy_dqn, thi_dqn = run_episode(env_dqn, agent.act, agent=agent, max_steps=episode_length, seed=trial_seed)
    energy_heuristic, thi_heuristic = run_episode(env_heuristic, heuristic_policy, max_steps=episode_length, seed=trial_seed)

    dqn_energies.append(energy_dqn)
    heuristic_energies.append(energy_heuristic)

    # 計算 DQN 策略中 THI 占比
    dqn_count_22_24 = sum(1 for val in thi_dqn if 22 <= val <= 24)
    dqn_count_19_27 = sum(1 for val in thi_dqn if 19 <= val <= 27)
    dqn_prop_22_24 = dqn_count_22_24 / len(thi_dqn) if len(thi_dqn) > 0 else 0
    dqn_prop_19_27 = dqn_count_19_27 / len(thi_dqn) if len(thi_dqn) > 0 else 0

    dqn_prop_22_24_list.append(dqn_prop_22_24)
    dqn_prop_19_27_list.append(dqn_prop_19_27)

    # 計算 Heuristic 策略中 THI 占比
    heuristic_count_22_24 = sum(1 for val in thi_heuristic if 22 <= val <= 24)
    heuristic_count_19_27 = sum(1 for val in thi_heuristic if 19 <= val <= 27)
    heuristic_prop_22_24 = heuristic_count_22_24 / len(thi_heuristic) if len(thi_heuristic) > 0 else 0
    heuristic_prop_19_27 = heuristic_count_19_27 / len(thi_heuristic) if len(thi_heuristic) > 0 else 0

    heuristic_prop_22_24_list.append(heuristic_prop_22_24)
    heuristic_prop_19_27_list.append(heuristic_prop_19_27)

    print(f"Trial {trial+1:2d}:")
    print(f"  DQN Energy = {energy_dqn:6.3f} kWh, Heuristic Energy = {energy_heuristic:6.3f} kWh")
    print(f"  DQN THI 占比 (22~24): {dqn_prop_22_24:.3f}, (19~27): {dqn_prop_19_27:.3f}")
    print(f"  Heuristic THI 占比 (22~24): {heuristic_prop_22_24:.3f}, (19~27): {heuristic_prop_19_27:.3f}\n")

# ---------------------------
# 統計分析
# ---------------------------
dqn_energies = np.array(dqn_energies)
heuristic_energies = np.array(heuristic_energies)
dqn_prop_22_24_list = np.array(dqn_prop_22_24_list)
dqn_prop_19_27_list = np.array(dqn_prop_19_27_list)
heuristic_prop_22_24_list = np.array(heuristic_prop_22_24_list)
heuristic_prop_19_27_list = np.array(heuristic_prop_19_27_list)

print("=== 能耗統計 ===")
print(f"{'DQN ':>10}平均耗電量: {np.mean(dqn_energies):.3f} kWh, 標準差: {np.std(dqn_energies):.3f}")
print(f"{'Heuristic ':.>10}平均耗電量: {np.mean(heuristic_energies):.3f} kWh, 標準差: {np.std(heuristic_energies):.3f}")
t_stat_energy, p_value_energy = stats.ttest_rel(dqn_energies, heuristic_energies)
print(f"配對 t 檢定（能耗）: t = {t_stat_energy:.3f}, p = {p_value_energy:.3f}\n")

print("=== THI 占比 (22~24) 統計 ===")
print(f"{'DQN ':>10}平均占比: {np.mean(dqn_prop_22_24_list):.3f}, 標準差: {np.std(dqn_prop_22_24_list):.3f}")
print(f"{'Heuristic ':.>10}平均占比: {np.mean(heuristic_prop_22_24_list):.3f}, 標準差: {np.std(heuristic_prop_22_24_list):.3f}")
t_stat_22_24, p_value_22_24 = stats.ttest_rel(dqn_prop_22_24_list, heuristic_prop_22_24_list)
print(f"配對 t 檢定 (22~24): t = {t_stat_22_24:.3f}, p = {p_value_22_24:.3f}\n")

print("=== THI 占比 (19~27) 統計 ===")
print(f"{'DQN ':>10}平均占比: {np.mean(dqn_prop_19_27_list):.3f}, 標準差: {np.std(dqn_prop_19_27_list):.3f}")
print(f"{'Heuristic ':.>10}平均占比: {np.mean(heuristic_prop_19_27_list):.3f}, 標準差: {np.std(heuristic_prop_19_27_list):.3f}")
t_stat_19_27, p_value_19_27 = stats.ttest_rel(dqn_prop_19_27_list, heuristic_prop_19_27_list)
print(f"配對 t 檢定 (19~27): t = {t_stat_19_27:.3f}, p = {p_value_19_27:.3f}\n")

# ---------------------------
# 可視化結果 (箱形圖)
# ---------------------------
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.boxplot([dqn_energies, heuristic_energies], tick_labels=["DQN", "Heuristic"])
plt.ylabel("Cumulative Energy (kWh)")
plt.title("Energy")

plt.subplot(2, 2, 2)
plt.boxplot([dqn_prop_22_24_list, heuristic_prop_22_24_list], tick_labels=["DQN", "Heuristic"])
plt.ylabel("Proportion of THI 22~24")
plt.title("Proportion of THI (22~24)")

plt.subplot(2, 2, 3)
plt.boxplot([dqn_prop_19_27_list, heuristic_prop_19_27_list], tick_labels=["DQN", "Heuristic"])
plt.ylabel("Proportion of THI 19~27")
plt.title("Proportion of THI (19~27)")

plt.tight_layout()
plt.show()
