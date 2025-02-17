import gym
from gym import spaces
import numpy as np
import pandas as pd

class AirConditioningEnv(gym.Env):
    def __init__(self, csv_path, render_mode=None):
        super(AirConditioningEnv, self).__init__()
        self.render_mode = render_mode

        # read CSV
        self.weather_data = pd.read_csv(csv_path)

        # make sure CSV included T_outside and AH
        if "氣溫(℃)" not in self.weather_data.columns or "AH(g/m³)" not in self.weather_data.columns:
            raise ValueError("CSV 檔案缺少必要的欄位 '氣溫(℃)' 和 'AH(g/m³)'！")


        # State space: indoor temperature, outdoor temperature, power consumption
        self.observation_space = spaces.Box(
            low=np.array([16, -1, 0], dtype=np.float32),  # Minimum ac temperature 16°C, outdoor minimum -1°C, minimum power consumption 0
            high=np.array([30, 40, 3000], dtype=np.float32),  # Maximum ac temperature 30°C, maximum outdoor 40°C, power consumption 3000 Wh
            dtype=np.float32
        )

        # Action space: [0: reduce ac temperature, 1: maintain, 2: increase ac temperature]
        self.action_space = spaces.Discrete(3)

        # Initialize environment variables
        self.T_ac = 25  # Initial ac temperature
        self.current_index = np.random.randint(0, 40000)  # 隨機起始點, To track the number of CSV data currently read
        self.T_outside = self.weather_data.iloc[self.current_index]["氣溫(℃)"]
        self.energy_consumption = 0  # Initial energy consumption

    # Calculate THI
    def calculate_THI(self, T, AH):
        Td = self.calculate_Td(T, AH)
        return T - 0.55 * (1 - np.exp((17.269 * Td) / (Td + 237.3)) / np.exp((17.269 * T) / (T + 237.3))) * (T - 14)

    # Calculate the indoor dew point temperature Td
    def calculate_Td(self, T, AH):
        RH = self.absolute_to_relative_humidity(T, AH)
        return T * RH / 100

    # Calculate power consumption
    def calculate_power_consumption(self, T_out, T_ac, Cooling_capacity=200):
        """ 當室外溫度小於冷氣溫度時冷氣關閉耗電量 0 """
        return Cooling_capacity * (T_out - T_ac) / T_ac if T_out > T_ac else 0

    # Min-Max Normalization
    def normalize(self, value, min_val, max_val):
        return (value - min_val) / (max_val - min_val)

    # Calculate absolute to relative humidity
    def absolute_to_relative_humidity(self, T, AH):
        """
        From absolute humidity AH(g/m³) and temperature T(°C) to calculate relative humidity RH(%)
        "New Equations for Computing Vapor Pressure and Enhancement Factor(Arden L. Buck)"
        eq: a * exp(b*T / (T+c))
        1. a is the saturated vapor pressure of water at different temperatures
        2. b and c is the empirical coefficient
        3. b and c use temperature inerval 0 ~ 50
        4. 217 is Correction of molar mass of water(Mw=18.015 g/mol)
        5. T + 273.15 is ℃ to K
        """
        # Calculate the Saturated water vapor pressure(hPa)
        Es = 6.1121 * np.exp((17.368 * T) / (T + 238.88))
        # Calculating the Actual water vapor pressure(hPa)
        Ea = (AH * (T + 273.15)) / 217
        # Calculating relative humidity(%)
        RH = ((Ea / Es) * 100).round(3)
        return RH

    # Calculate Reward
    def calculate_reward(self, AH, T_out, T_ac, a=1, b=0.1, THI_optimal=23, THI_min=0, THI_max=35, Power_min=0, Power_max=3000):

        THI = self.calculate_THI(T_ac, AH)
        PowerConsumption = self.calculate_power_consumption(T_out, T_ac)

        # normalize
        THI_norm = self.normalize(THI, THI_min, THI_max)
        Power_norm = self.normalize(PowerConsumption, Power_min, Power_max)
        THI_optimal_norm = self.normalize(THI_optimal, THI_min, THI_max)

        # Calculate Reward
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
        reward, THI, PowerConsumption = self.calculate_reward(AH, self.T_outside, self.T_ac)

        # 計算耗電量
        self.energy_consumption += PowerConsumption

        # 環境自然終止條件 self.energy_consumption > 3000 or
        done = self.current_index >= len(self.weather_data)

        # 回傳新的狀態
        state = np.array([self.T_ac, self.T_outside, self.energy_consumption], dtype=np.float32)

        info = {"THI": THI, "power": PowerConsumption}

        return state, reward, done, info

    def reset(self, seed=None, options=None, return_info=False):
        """ 重設環境並從第一筆天氣數據開始 """
        if seed is not None:
          # 設定 numpy 隨機種子，你也可以根據需要設定其他隨機數生成器
          np.random.seed(seed)
        self.T_ac = 25
        self.energy_consumption = 0
        self.current_index = np.random.randint(0, 40000)
        self.T_outside = self.weather_data.iloc[self.current_index]["氣溫(℃)"]

        # 回傳初始狀態 (室內溫度, 室外溫度, 累計耗電)
        state = np.array([self.T_ac, self.T_outside, self.energy_consumption], dtype=np.float32)
        if return_info:
            return state, {}  # 回傳一個空的 info 字典，或是你想要的其他資訊
        return state

    def render(self, mode="human"):
        """ 可選的環境視覺化方法 """
        print(f"Outside Temp: {self.T_outside:.2f}°C, AC Temp: {self.T_ac}, Energy: {self.energy_consumption:.2f} Wh")
