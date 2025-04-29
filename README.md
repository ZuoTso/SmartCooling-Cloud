# 智慧空調控制系統 (Smart Cooling: 深度強化學習 DRL)

## 專案簡介

- 本專案透過深度強化學習 (DQN) 建立智慧空調控制系統，實現能耗與舒適度的平衡調控。
- 透過[CODiS 氣候觀測資料查詢服務](https://codis.cwa.gov.tw/)使用真實天氣資料
- 依需求調整 `a`, `b` 參數以改變舒適度與節能權重。
- 以氣象署使用的Temperature–Humidity Index（THI）作為舒適度評估指標。
- Heuristic 為政府公家機關使用冷氣的規範，超過28度C開冷氣，溫度設定在26~28度C

## 功能特色

- 基於 DQN 的策略學習
- 優先經驗回放 (Prioritized Experience Replay)
- 自訂化獎勵函數 (THI 舒適度與能源消耗)
- 模擬真實環境資料 (weather data)
- 可比較 Heuristic 與 DQN 策略表現

## 專案結構

```bash
AirConditioning/
├── get data program                      # 為raw data 獲取的方式
├── DQN.py                                # DQN and Prioritized Experience Replay Architecture
├── Main-train and plot.py                # Main：Training DQN agent
├── performance.py                        # 測試腳本：比較 DQN 與 Heuristic 策略 (Smartcooling.ipynb 中有多次檢測)
├── air_conditioning_env.py               # 環境定義 (Gym Env)
├── THI reward function.py                # 獎勵函數實作範例
├── NormalizeObservation wapper.py        # 觀察值標準化 wrapper
├── raw data                              
├── training data/training_data_2_11.csv  # 訓練資料 (weather data)
├── test data/test data_2_11.csv          # 測試資料
└── README.md                             # 專案說明
```

## 安裝與依賴

### requirements

```
torch
gym
numpy
pandas
matplotlib
psychrolib
```

## 獎勵函數設定

詳細實作於 `air_conditioning_env.py`，可依需求調整 `a`, `b` 參數以改變舒適度與節能權重。

## 資料準備

- `training_data_2_11.csv`：用於模型訓練，包含氣溫、絕對濕度等欄位。
- `test data_2_11.csv`：用於策略比較與評估。

## 未來改進

- 探索 Rainbow DQN 等進階演算法
- 擴展為 Multi-Zone 控制模型
- 結合時段電價與使用者回饋
