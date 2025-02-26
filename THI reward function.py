#%%
def calculate_reward(self, AH, T_out, T_in, a=0.9, b=0.1, THI_optimal=23, THI_min=0, THI_max=35, Power_min=0, Power_max=3, k=10):
        # 計算 THI 與 PowerConsumption
        THI = self.calculate_THI(T_in, AH)
        PowerConsumption = self.calculate_power_consumption(T_out, T_in)
        Power_norm = self.normalize(PowerConsumption, Power_min, Power_max)
        
        # 以下為新的 THI 部分參數設定
        A = 1.0       # 基本懲罰強度（僅用於 THI >= 23 的部分）
        beta = 0.2    # 指數下降速率
        B = 0.3       # bonus 最大值
        sigma = 0.7   # 高斯 bonus 衰減參數
        delta = 0.2   # 在 THI=23 附近的混合區半寬

        # 當 THI < 23 時，採用線性遞減，保證 THI=23 時 reward=0，THI=THI_min 時 reward=-1
        def linear_reward(THI_val):
            return - (THI_optimal - THI_val) / (THI_optimal - THI_min)

        # 當 THI >= 23 時，採用指數下降，並正規化使得 THI=THI_max 時 reward=-1
        def exponential_reward(THI_val):
            norm = A * (np.exp(beta * (THI_max - THI_optimal)) - 1)
            return -A * (np.exp(beta * (THI_val - THI_optimal)) - 1) / norm

        # 平滑混合函數：使用 smoothstep 函數（3t² - 2t³）
        def smoothstep(t):
            return 3 * t**2 - 2 * t**3

        # 在 THI<23‑δ 時使用線性函數，THI>23+δ 時使用指數函數，在 [23‑δ, 23+δ] 區間內平滑混合兩者
        def base_reward(THI_val):
            if THI_val <= THI_optimal - delta:
                return linear_reward(THI_val)
            elif THI_val >= THI_optimal + delta:
                return exponential_reward(THI_val)
            else:
                t = (THI_val - (THI_optimal - delta)) / (2 * delta)
                w = smoothstep(t)
                return (1 - w) * linear_reward(THI_val) + w * exponential_reward(THI_val)

        # 在 THI 附近（22～24）使用 Gaussian bonus，當 THI=23 時 bonus = B
        def bonus_reward(THI_val):
            return B * np.exp(-((THI_val - THI_optimal)**2) / (2 * sigma**2))

        # 結合 base reward 與 bonus（未正規化）
        THI_reward_unscaled = base_reward(THI) + bonus_reward(THI)

        # 以離散取樣計算整個 THI 定義域內的最小與最大值，用以正規化
        THI_values = np.linspace(THI_min, THI_max, 400)
        combined_vals = np.array([base_reward(val) + bonus_reward(val) for val in THI_values])
        r_min = np.min(combined_vals)
        r_max = np.max(combined_vals)

        # 正規化：將 THI_reward_unscaled 映射到 [-1, 0] 區間
        THI_reward = (THI_reward_unscaled - r_min) / (r_max - r_min) - 1

        # 最終 reward 結合 THI 與 Power 的部分權重
        reward = a * THI_reward - b * Power_norm
        return reward, THI, PowerConsumption
#%%
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
#%%
def calculate_reward(self, AH, T_out, T_in, a=0.9, b=0.1, k=10, THI_optimal=23, Power_min=0, Power_max=3):
        """
        修改後的 calculate_reward：
          1. 利用 self.calculate_THI(T_in, AH) 計算 THI
          2. 計算 PowerConsumption 及其正規化值 Power_norm
          3. 利用 smooth_reward 函數計算 THI_reward，其中：
            - THI < 10 或 THI > 30 時，reward = 0
            - THI 在 10～15 或 27～30 時，reward = 0.05
            - THI 在 15～19 時，reward = 0.1
            - THI 在 19～22 或 24～27 時，reward = 0.6
            - THI 在 22～24 時，reward = 1
            此函數利用多個 sigmoid 函數在臨界點處平滑過渡
          4. 最終 reward = a * THI_reward - b * Power_norm
        """
        THI = self.calculate_THI(T_in, AH)
        PowerConsumption = self.calculate_power_consumption(T_out, T_in)
        Power_norm = self.normalize(PowerConsumption, Power_min, Power_max)

        # 定義平滑的 THI reward 函數
        def smooth_reward(THI, k=10):
            def S(t):
                return 1 / (1 + np.exp(-k * (THI - t)))
            return 0.05 * S(10) + 0.05 * S(15) + 0.5 * S(19) + 0.4 * S(22) - 0.4 * S(24) - 0.55 * S(27) - 0.05 * S(30)

        THI_reward = smooth_reward(THI, k)

        reward = a * THI_reward - b * Power_norm
        return reward, THI, PowerConsumption
