import pandas as pd
import numpy as np

def calculate_THI(df):
    df['露點溫度(℃)'] = pd.to_numeric(df['露點溫度(℃)'], errors='coerce')
    df['氣溫(℃)'] = pd.to_numeric(df['氣溫(℃)'], errors='coerce')
    Td = df['露點溫度(℃)']
    T = df['氣溫(℃)']
    THI = T - 0.55 * (1 - np.exp((17.269 * Td) / (Td + 237.3)) / np.exp((17.269 * T) / (T + 237.3))) * (T - 14)
    df["THI"] = THI.round(2)
    return df

csv_path = "training data.csv"
df = pd.read_csv(csv_path)
df = calculate_THI(df)
print(max(df["THI"]))
print(min(df["THI"]))
print(max(df["氣溫(℃)"]))
print(min(df["氣溫(℃)"]))