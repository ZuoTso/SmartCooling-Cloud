import pandas as pd

def fix_hour(time_str):
    """
    處理時間字串，當發現小時部分為 "24" 時，
    將日期往後推一天並將小時設為 "00"。
    """
    time_str = time_str.strip()  # 去除左右空白
    parts = time_str.split("-")  # 預期格式為 YYYY-MM-DD-HH
    if parts[-1] == "24":
        # 若小時為 24，先取出年月日部分
        date_part = "-".join(parts[:-1])
        # 解析年月日部分
        dt = pd.to_datetime(date_part, format="%Y-%m-%d")
        # 加一天並設為午夜 00
        dt += pd.Timedelta(days=1)
        # 重新組合字串，格式維持為 "YYYY-MM-DD-00"
        return dt.strftime("%Y-%m-%d-00")
    else:
        return time_str

csv_file = "test data/test data.csv"
df = pd.read_csv(csv_file)

# 先處理「觀測時間(hour)」欄位：移除空白並處理可能的 "24" 小時問題
df["觀測時間(hour)"] = df["觀測時間(hour)"].astype(str).apply(fix_hour)

df["觀測時間(hour)"] = pd.to_datetime(df["觀測時間(hour)"], format="%Y-%m-%d-%H", errors="coerce")

# 檢查是否有解析失敗的資料
if df["觀測時間(hour)"].isna().any():
    print("以下資料無法解析，請檢查格式：")
    print(df[df["觀測時間(hour)"].isna()]["觀測時間(hour)"])
    raise ValueError("日期格式解析錯誤，請確認 '觀測時間(hour)' 欄位格式是否為 'YYYY-MM-DD-HH'。")

# 提取月份資訊
df["month"] = df["觀測時間(hour)"].dt.month

# 定義各月份區間的遮罩條件
mask_2_6    = df["month"].isin([2, 3, 4, 5, 6])
mask_7_9    = df["month"].isin([7, 8, 9])
mask_10_11  = df["month"].isin([10, 11])
mask_12_1   = df["month"].isin([12, 1])
mask_2_11   = df["month"].isin([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

# 分別篩選各區間的資料
df_2_6   = df[mask_2_6]
df_7_9   = df[mask_7_9]
df_10_11 = df[mask_10_11]
df_12_1  = df[mask_12_1]
df_2_11  = df[mask_2_11]

# 儲存成不同的 CSV 檔案
# df_2_6.to_csv("training_data_2_6.csv", index=False)
# df_7_9.to_csv("training_data_7_9.csv", index=False)
# df_10_11.to_csv("training_data_10_11.csv", index=False)
# df_12_1.to_csv("training_data_12_1.csv", index=False)
# df_2_11.to_csv("training_data_2_11.csv", index=False)

df_2_6.to_csv("test data_2_6.csv", index=False)
df_7_9.to_csv("test data_7_9.csv", index=False)
df_10_11.to_csv("test data_10_11.csv", index=False)
df_12_1.to_csv("test data_12_1.csv", index=False)
df_2_11.to_csv("test data_2_11.csv", index=False)

print("資料已依照月份區間拆分成 CSV 檔案。")
