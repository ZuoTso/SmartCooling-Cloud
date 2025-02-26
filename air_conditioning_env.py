import gym
from gym import spaces
import numpy as np
import pandas as pd
import math

class AirConditioningEnv(gym.Env):
    def __init__(self, csv_path, episode_length=168, render_mode=None):
        super(AirConditioningEnv, self).__init__()
        self.render_mode = render_mode
        self.episode_length = episode_length  # The number of steps per episode (e.g. 24 for one day)
        self.step_counter = 0

        # read CSV
        self.weather_data = pd.read_csv(csv_path)

        # make sure CSV included T_outside and AH
        if "氣溫(℃)" not in self.weather_data.columns or "AH(g/m³)" not in self.weather_data.columns:
            raise ValueError("CSV 檔案缺少必要的欄位 '氣溫(℃)' 和 'AH(g/m³)'！")


        # State space: indoor temperature, outdoor temperature, power consumption
        self.observation_space = spaces.Box(
            # Minimum ac temperature 16°C, outdoor -1°C, power consumption 0 kWh, THI 0, indoor -1°C
            low=np.array([16, -1, 0, 0, -1], dtype=np.float32),
            # Maximum ac temperature 30°C, outdoor 40°C, power consumption 3 kWh, THI 35, indoor 40°C
            high=np.array([30, 40, 3, 35, 40], dtype=np.float32),
            dtype=np.float32
        )

        # Action space: [0: reduce ac temperature, 1: maintain, 2: increase ac temperature]
        self.action_space = spaces.Discrete(3)

        # Initialize environment variables
        self.T_ac = 25  # Initial ac temperature
        self.current_index = np.random.randint(0, len(self.weather_data) - self.episode_length) # Random starting point, To track the number of CSV data currently read
        self.T_outside = self.weather_data.iloc[self.current_index]["氣溫(℃)"]
        self.T_in = self.T_outside  # Initial indoor temperature is the same as the outside temperature
        self.energy_consumption = 0  # Initial energy consumption

    # Calculate THI
    def calculate_THI(self, T, AH):
        Td = self.calculate_Td(T, AH)
        THI = T - 0.55 * (1 - np.exp((17.269 * Td) / (Td + 237.3)) / np.exp((17.269 * T) / (T + 237.3))) * (T - 14)
        if np.isnan(THI):
            print("NaN detected in env THI!")
            print(f"THI_norm: {THI}, T: {T}, AH: {AH}")
        return THI
    # Calculate the indoor dew point temperature Td
    def calculate_Td(self, T, AH):
        RH = self.absolute_to_relative_humidity(T, AH)
        return T * RH / 100

    # Calculate power consumption
    def calculate_power_consumption(self, T_out, T_in, Cooling_capacity=2):
        """
        When the outdoor temperature is lower than the air conditioning temperature,
        the air conditioning is turned off,
        and the power consumption : 0
        """
        if T_in == 0:
          return 0
        if T_out > T_in:
          PowerConsumption = Cooling_capacity * (T_out - T_in) / T_in
        else:
          PowerConsumption = 0
        return PowerConsumption
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
    def calculate_reward(self, AH, T_out, T_in, a=0.9, b=0.1, THI_optimal=23, THI_min=0, THI_max=35, Power_min=0, Power_max=3, k=10):
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
        4. Basic THI penalty, using a logarithmic function: the larger the deviation, the heavier the penalty (the lower the reward value)
        5. Use sigmoid to generate extra rewards in the range of 20~26,
          the coefficient is adjustable (here 0.3 is the maximum bonus).
          Additional bonus in the range of 22~24 (here 0.2 is the maximum additional bonus)

        """
        THI = self.calculate_THI(T_in, AH)
        PowerConsumption = self.calculate_power_consumption(T_out, T_in)

        # normalize
        Power_norm = self.normalize(PowerConsumption, Power_min, Power_max)

        # Basic THI penalty
        base_reward = -math.log(1 + abs(THI - THI_optimal))

        # Define the sigmoid function to smooth the transition
        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        bonus_range = 0.1 * (sigmoid(k * (THI - 20)) - sigmoid(k * (THI - 26)))
        extra_bonus = 0.6 * (sigmoid(k * (THI - 22)) - sigmoid(k * (THI - 24)))

        bonus = bonus_range + extra_bonus

        THI_reward = base_reward + bonus

        # Calculate Reward
        reward = a * THI_reward - b * Power_norm
        return reward, THI, PowerConsumption

    def step(self, action):
        """ Update the environment based on the action and calculate the reward """

        # Get the CSV external temperature and absolute humidity at the corresponding time
        self.T_outside = self.weather_data.iloc[self.current_index]["氣溫(℃)"]
        AH = self.weather_data.iloc[self.current_index]["AH(g/m³)"]
        self.current_index += 1  # Update the index and enter the next hour
        self.step_counter += 1

        # Movement affects ac temperature
        if action == 0:  # Lower the ac temperature
            self.T_ac = max(16, self.T_ac - 1)
        elif action == 2:  # Increase ac temperature
            self.T_ac = min(30, self.T_ac + 1)

        # Determine the indoor temperature
        if self.T_ac < self.T_outside:
            self.T_in = self.T_ac  # The ac is on, indoor temperature equals to the ac set temperature
        else:
            self.T_in = self.T_outside  # ac is off, indoor temperature equals outdoor temperature

        # Calculating reward
        reward, THI, PowerConsumption = self.calculate_reward(AH, self.T_outside, self.T_in)

        # Calculate power consumption
        self.energy_consumption += PowerConsumption

        # Environmental natural termination conditions
        done = self.step_counter >= self.episode_length or self.current_index >= len(self.weather_data)

        # New state
        state = np.array([self.T_ac, self.T_outside, PowerConsumption, THI, self.T_in], dtype=np.float32)

        info = {"THI": THI, "energy": self.energy_consumption}

        return state, reward, done, info

    def reset(self, seed=None, options=None, return_info=False):
        if seed is not None:
          # Set numpy random seed
          np.random.seed(seed)
        self.T_ac = 25
        self.energy_consumption = 0
        PowerConsumption = 0
        self.step_counter = 0
        self.current_index = np.random.randint(0, len(self.weather_data) - self.episode_length)
        self.T_outside = self.weather_data.iloc[self.current_index]["氣溫(℃)"]
        AH = self.weather_data.iloc[self.current_index]["AH(g/m³)"]

        if self.T_ac < self.T_outside:
            self.T_in = self.T_ac
        else:
            self.T_in = self.T_outside

        THI = self.calculate_THI(self.T_ac, AH)

        # Return initial status
        state = np.array([self.T_ac, self.T_outside, PowerConsumption, THI, self.T_in], dtype=np.float32)
        if return_info:
            return state, {}
        return state

    def render(self, mode="human"):
        """ Alternative methods of visualizing the environment """
        print(f"Outside Temp: {self.T_outside:.2f}°C, AC Temp: {self.T_ac}, Energy: {self.energy_consumption:.2f} Wh")
