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
            low=np.array([16, -1, 0, 0], dtype=np.float32),  # Minimum ac temperature 16°C, outdoor -1°C, power consumption 0, THI 0
            high=np.array([30, 40, 3000, 50], dtype=np.float32),  # Maximum ac temperature 30°C, outdoor 40°C, power consumption 3000 Wh, THI 50
            dtype=np.float32
        )

        # Action space: [0: reduce ac temperature, 1: maintain, 2: increase ac temperature]
        self.action_space = spaces.Discrete(3)

        # Initialize environment variables
        self.T_ac = 25  # Initial ac temperature
        self.current_index = np.random.randint(0, 40000)  # Random starting point, To track the number of CSV data currently read
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
        """
        When the outdoor temperature is lower than the air conditioning temperature,
        the air conditioning is turned off,
        and the power consumption : 0
        """
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
        """
        1. a and b are weight coefficients that tend to optimize power consumption or comfort(THI)
        2. THI table
          -----------
          <10 | verycold
          10-15 | cold
          16-19 | a little cold
          20-26 | comfortable
          27-30 | hot
          >30 | very hot
          -----------
        3. When indoor comfort is close to ideal and power consumption is low,
          the overall negative reward becomes smaller (i.e., better performance)

        """
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
        """ Update the environment based on the action and calculate the reward """
        # Get the CSV external temperature and absolute humidity at the corresponding time
        self.T_outside = self.weather_data.iloc[self.current_index]["氣溫(℃)"]
        AH = self.weather_data.iloc[self.current_index]["AH(g/m³)"]
        self.current_index += 1  # Update the index and enter the next hour

        # Movement affects ac temperature
        if action == 0:  # Lower the ac temperature
            self.T_ac = max(16, self.T_ac - 1)
        elif action == 2:  # Increase ac temperature
            self.T_ac = min(30, self.T_ac + 1)

        # Indoor temperature changes, assuming that it will reach the cooling temperature within one hour (simplified calculation)

        # Calculating reward
        reward, THI, PowerConsumption = self.calculate_reward(AH, self.T_outside, self.T_ac)

        # Calculate power consumption
        self.energy_consumption += PowerConsumption

        # Environmental natural termination conditions self.energy_consumption > 3000 or
        done = self.current_index >= len(self.weather_data)

        # New state
        state = np.array([self.T_ac, self.T_outside, self.energy_consumption, THI], dtype=np.float32)

        info = {"THI": THI, "power": PowerConsumption}

        return state, reward, done, info

    def reset(self, seed=None, options=None, return_info=False):
        if seed is not None:
          # Set numpy random seed
          np.random.seed(seed)
        self.T_ac = 25
        self.energy_consumption = 0
        self.current_index = np.random.randint(0, 40000)
        self.T_outside = self.weather_data.iloc[self.current_index]["氣溫(℃)"]
        THI = 23

        # Return initial status
        state = np.array([self.T_ac, self.T_outside, self.energy_consumption, THI], dtype=np.float32)
        if return_info:
            return state, {}
        return state

    def render(self, mode="human"):
        """ Alternative methods of visualizing the environment """
        print(f"Outside Temp: {self.T_outside:.2f}°C, AC Temp: {self.T_ac}, Energy: {self.energy_consumption:.2f} Wh")
