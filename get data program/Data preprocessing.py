import pandas as pd
import numpy as np
import os
import re

# Set folder path
folder_path = "raw data/test"
csv_files = [file for file in os.listdir(folder_path) if file.endswith(".csv")]

# 提取的欄位名稱
# , '測站氣壓(hPa)', '海平面氣壓(hPa)'
columns_to_extract = ['觀測時間(hour)', '氣溫(℃)', '露點溫度(℃)', '相對溼度(%)']

# Read, Merge and extract all CSV
def columns_extract(columns_to_extract):
    dfs = []
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        # 移除每個檔案的標題行
        df = df[df.index != 0].reset_index(drop=True)
        df = df[columns_to_extract]

        # 根據檔名提取觀測時間
        # 假設檔名格式為 "YYYYMMDD_其他資訊.csv"，例如 "20230201_observation.csv"
        match = re.search(r"-(\d{4}-\d{2}-\d{2})\.csv$", file)
        if match:
            observation_time = match.group(1)

            # 將觀測時間加入資料框中
            df['觀測時間(hour)'] = observation_time + '-' + df['觀測時間(hour)']
        else:
            observation_time = None  # 若無法從檔名中取得，可設為 None 或給予預設值
            print(file)
        
        dfs.append(df)
        extracted_df = pd.concat(dfs, ignore_index=True)
    return extracted_df

def calculate_absolute_humidity(df):
    """
    "New Equations for Computing Vapor Pressure and Enhancement Factor(Arden L. Buck)"
    用 Arden Buck equations 由氣溫(℃)和露點溫度(℃) 計算絕對溼度 AH (g/m³)
    eq: a * exp(b*T / (T+c))
    1. a 為水在不同溫度時的飽和蒸汽壓
    2. b and c 是經驗係數
    3. b and c use temperature inerval 0 ~ 50
    4. 217 是 Mw 水的摩爾質量(18.015 g/mol)的修正
    5. T + 273.15 is ℃ to K
    """

    df['露點溫度(℃)'] = pd.to_numeric(df['露點溫度(℃)'], errors='coerce')
    df['氣溫(℃)'] = pd.to_numeric(df['氣溫(℃)'], errors='coerce')
    Td = df['露點溫度(℃)']
    T = df['氣溫(℃)']
    # 計算實際水蒸氣壓力(hPa)
    Ea = 6.1121 * np.exp((17.368 * Td) / (Td + 238.88))
    # 計算絕對溼度(g/m³)
    AH = (217 * Ea) / (T + 273.15)
    df['AH(g/m³)'] = AH.round(3)
    return df

df = columns_extract(columns_to_extract)
df = df.dropna().replace('X', np.nan).dropna()
df = calculate_absolute_humidity(df)
df.to_csv("test data.csv", index=False)
