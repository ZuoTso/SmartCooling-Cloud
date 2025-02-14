import gym
from gym import spaces
import numpy as np
import pandas as pd

class AirConditioningEnv(gym.Env):
    def __init__(self, csv_path):
        super(AirConditioningEnv, self).__init__()

        # 讀取 CSV 檔案
        self.weather_data = pd.read_csv(csv_path)

        # 確保 CSV 包含 T_outside 和 AH
        if "氣溫(℃)" not in self.weather_data.columns or "AH(g/m³)" not in self.weather_data.columns:
            raise ValueError("CSV 檔案缺少必要的欄位 '氣溫(℃)' 和 'AH(g/m³)'！")


        # 狀態空間：室內溫度, 室外溫度, 冷氣強度, 耗電量
        self.observation_space = spaces.Box(
            low=np.array([16, -1, 0]),  # 溫度最低 16°C，室外最低 -1°C，耗電最小 0
            high=np.array([30, 40, 3000]),  # 溫度最高 30°C，室外最高 40°C，累計耗電 3000Wh
            dtype=np.float32
        )

        # 動作空間: [0: 降低冷氣溫度, 1: 保持, 2: 增加冷氣溫度]
        self.action_space = spaces.Discrete(3)

        # 初始化環境變數
        self.T_ac = 25  # 初始溫度
        self.current_index = 0  # 用來追蹤當前讀取到 CSV 的第幾筆數據
        self.T_outside = self.weather_data.iloc[self.current_index]["氣溫(℃)"]
        self.energy_consumption = 0  # 初始能耗

    # 計算 THI
    def calculate_THI(self, T, AH):
        Td = self.calculate_Td(T, AH)
        return T - 0.55 * (1 - np.exp((17.269 * Td) / (Td + 237.3)) / np.exp((17.269 * T) / (T + 237.3))) * (T - 14)

    # 計算耗電量
    def calculate_power_consumption(self, T_out, T_ac, Cooling_capacity=200):
        """ 當室外溫度小於冷氣溫度時冷氣關閉耗電量 0 """
        return Cooling_capacity * (T_out - T_ac) / T_ac if T_out > T_ac else 0

    # Min-Max Normalization
    def normalize(self, value, min_val, max_val):
        return (value - min_val) / (max_val - min_val)

    # 計算絕對溼度轉相對濕度
    def absolute_to_relative_humidity(self, T, AH):
        """
        由絕對濕度 AH (g/m³) 和氣溫 T (°C) 計算相對濕度 RH (%)
        "New Equations for Computing Vapor Pressure and Enhancement Factor(Arden L. Buck)"
        eq: a * exp(b*T / (T+c))
        1. a 為水在不同溫度時的飽和蒸汽壓
        2. b and c 是經驗係數
        3. b and c use temperature inerval 0 ~ 50
        4. 217 是 Mw 水的摩爾質量(18.015 g/mol)的修正
        5. T + 273.15 is ℃ to K
        """
        # 計算飽和水蒸氣壓力（hPa）
        Es = 6.1121 * np.exp((17.368 * T) / (T + 238.88))
        # 計算實際水蒸氣壓力（hPa）
        Ea = (AH * (T + 273.15)) / 217
        # 計算相對濕度（%）
        RH = ((Ea / Es) * 100).round(3)
        return RH

    # 計算 室內露點溫度 Td
    def calculate_Td(self, T, AH):
        RH = self.absolute_to_relative_humidity(T, AH)
        return T * RH / 100

    # 計算 Reward
    def calculate_reward(self, T_out, T_ac, a=1, b=0.1, THI_optimal=23, THI_min=0, THI_max=35, Power_min=0, Power_max=3000):
        
        THI = self.calculate_THI(T_ac, self.AH)
        PowerConsumption = self.calculate_power_consumption(T_out, T_ac)
        
        # 正規化
        THI_norm = self.normalize(THI, THI_min, THI_max)
        Power_norm = self.normalize(PowerConsumption, Power_min, Power_max)
        THI_optimal_norm = self.normalize(THI_optimal, THI_min, THI_max)

        # 計算 Reward
        reward = - a * abs(THI_norm - THI_optimal_norm) - b * Power_norm
        return reward, THI, PowerConsumption


    def step(self, action):
        """ 根據動作更新環境並計算回報 """
        # 取得對應時間的外部溫度 & 絕對濕度
        self.T_outside = self.weather_data.iloc[self.current_index]["氣溫(℃)"]
        AH = self.weather_data.iloc[self.current_index]["AH(g/m³)"]
        self.current_index += 1  # 更新索引，進入下一個時間點

        # 動作影響冷氣溫度
        if action == 0:  # 降低冷氣溫度
            self.T_ac = max(16, self.T_ac - 1)
        elif action == 2:  # 增加冷氣溫度
            self.T_ac = min(30, self.T_ac + 1)

        # 室內溫度變化，假設為一小時以內會達到冷氣溫度(簡化計算)

        # 計算回報
        reward, THI, PowerConsumption = self.calculate_reward(self.T_outside, self.T_ac)

        # 計算耗電量
        self.energy_consumption += PowerConsumption
        
        # 環境終止條件 self.energy_consumption > 3000 or
        done = self.current_index >= len(self.weather_data)

        # 回傳新的狀態
        state = np.array([self.T_ac, self.T_outside, self.energy_consumption], dtype=np.float32)

        return state, reward, done, {}

    def reset(self):
        """ 重設環境並從第一筆天氣數據開始 """
        self.T_ac = 25
        self.energy_consumption = 0
        self.current_index = 0  # 重新從 CSV 第一筆數據開始
        self.T_outside = self.weather_data.iloc[self.current_index]["氣溫(℃)"]
        self.AH = self.weather_data.iloc[self.current_index]["AH(g/m³)"]

    def render(self, mode="human"):
        """ 可選的環境視覺化方法 """
        print(f"Outside Temp: {self.T_outside:.2f}°C, AC Temp: {self.T_ac}, Energy: {self.energy_consumption:.2f} Wh")
