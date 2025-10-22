import pandas as pd
import os # 匯入 os 模組來處理檔案路徑

# --- 1. CSV 檔案名稱 ---
csv_file_name = 'data_org.csv'

# --- 2. 時間欄位名稱 ---
time_column_name = 'DateTime'

# --- 3. 主要的時間段與標籤列表 ---
# 這裡包含您所有的時間定義
time_periods_with_labels = [
    # ('2025-04-11 09:14:00', '2025-04-11 09:44:00', 0),#冷凝盤管阻塞20%
    ('2025-04-11 09:46:00', '2025-04-11 10:16:00', 0),#冷凝盤管阻塞30%
    ('2025-04-11 10:29:00', '2025-04-11 10:59:00', 1),#蒸發盤管阻塞10%
    ('2025-04-11 11:01:00', '2025-04-11 11:31:00', 1),#蒸發盤管阻塞20%
    ('2025-04-11 11:33:00', '2025-04-11 12:03:00', 1),#蒸發盤管阻塞23%
    ('2025-04-11 12:05:00', '2025-04-11 13:35:00', 1),#正常資料常溫34度
    ('2025-04-11 13:45:00', '2025-04-11 14:15:00', 1),#蒸發風扇電流90%
    ('2025-04-11 14:19:00', '2025-04-11 14:49:00', 1),#蒸發風扇電流80%
    ('2025-04-11 14:52:00', '2025-04-11 15:22:00', 1),#蒸發風扇電流70%
    ('2025-04-11 15:46:00', '2025-04-11 16:16:00', 1),#正常資料高溫42.7度
    ('2025-04-14 09:00:00', '2025-04-14 10:00:00', 1),#正常資料低溫24度
    # ('2025-04-14 10:49:00', '2025-04-14 11:45:00', 1),#加熱器運轉
    ('2025-04-14 13:25:00', '2025-04-14 14:25:00', 1),#冷媒洩漏10%
    ('2025-04-14 14:50:00', '2025-04-14 15:50:00', 1),#冷媒洩漏20%
    # ('2026-01-01 00:00:00', '2026-01-01 00:30:00', 0),#壓縮機故障10%
    # ('2026-01-01 00:30:00', '2026-01-01 01:00:00', 0),#壓縮機故障20%
    # ('2026-01-01 01:00:00', '2026-01-01 01:30:00', 0),#壓縮機故障30%
    # ('2026-01-01 02:00:00', '2026-01-01 02:30:00', 0),#冷凝風扇電流上升10%
    # ('2026-01-01 02:30:00', '2026-01-01 03:00:00', 0),#冷凝風扇電流上升20%
    # ('2026-01-01 03:00:00', '2026-01-01 03:30:00', 0),#冷凝風扇電流上升30%
    # ('2026-01-01 04:00:00', '2026-01-01 04:30:00', 0),#冷凝風扇電流上升10%
    # ('2026-01-01 04:30:00', '2026-01-01 05:00:00', 0),#冷凝風扇電流下降20%
    # ('2026-01-01 05:00:00', '2026-01-01 05:30:00', 0),#冷凝風扇電流下降30%
    # ('2026-01-01 06:00:00', '2026-01-01 06:30:00', 0),#蒸發風扇電流上升10%
    # ('2026-01-01 06:30:00', '2026-01-01 07:00:00', 0),#蒸發風扇電流上升20%
    # ('2026-01-01 07:00:00', '2026-01-01 07:30:00', 0),#蒸發風扇電流上升30%
    # ('2026-01-01 08:00:00', '2026-01-01 08:30:00', 0),#蒸發風扇電流下降10%
    # ('2026-01-01 08:30:00', '2026-01-01 09:00:00', 0),#蒸發風扇電流下降20%
    # ('2026-01-01 09:00:00', '2026-01-01 09:30:00', 0),#蒸發風扇電流下降30%
    # ('2026-01-01 10:00:00', '2026-01-01 10:30:00', 0),#加熱器效率不良10%
    # ('2026-01-01 10:30:00', '2026-01-01 11:00:00', 0),#加熱器效率不良20%
    # ('2026-01-01 11:00:00', '2026-01-01 11:30:00', 0),#加熱器效率不良30%
]

# --- 4. 定義您想「分割出去」的特殊時間段 ---
# 將您想另外存檔的時間段完整複製到這裡
special_periods_to_extract = [
    ('2025-04-11 09:14:00', '2025-04-11 09:44:00', 0),#冷凝盤管阻塞20%
    ('2025-04-11 15:46:00', '2025-04-11 16:16:00', 1),
    ('2025-04-14 09:00:00', '2025-04-14 10:00:00', 1),
]

# --- 5. 請修改這裡：設定輸出的資料夾路徑 ---
# Windows 範例: 'C:/Users/YourUser/Desktop/output_data' (請務必使用正斜線 /)
# Mac/Linux 範例: '/Users/YourUser/Documents/output_data'
# 如果留空 ('')，檔案會儲存在和這個 Python 程式一樣的資料夾裡
output_directory = 'D:/DACAD/datasets/HVAC/' # <-- 請在這裡填寫您想儲存的完整路徑


# --- 主程式開始 (以下部分通常不需要修改) ---

# 讀取 CSV 檔案
try:
    df = pd.read_csv(csv_file_name)
except FileNotFoundError:
    print(f"錯誤：找不到名為 '{csv_file_name}' 的檔案。")
    print("請確認：1. 檔名是否拼寫正確 2. 此程式碼檔案和您的 CSV 檔是否在同一個資料夾中。")
    exit()

# 檢查時間欄位是否存在
if time_column_name not in df.columns:
    print(f"錯誤：在您的 CSV 檔案中找不到名為 '{time_column_name}' 的欄位。")
    print(f"檔案中包含的欄位有：{list(df.columns)}")
    exit()

# 將時間欄位轉換為 pandas 的 datetime 格式
df[time_column_name] = pd.to_datetime(df[time_column_name], errors='coerce')
df.dropna(subset=[time_column_name], inplace=True)


# --- 建立一個輔助函式來處理資料篩選與標記 ---
def process_periods(dataframe, periods_list):
    """根據給定的時間段列表，從 dataframe 中篩選、標記並合併資料"""
    labeled_dfs = []
    for start_str, end_str, label in periods_list:
        start_time = pd.to_datetime(start_str)
        end_time = pd.to_datetime(end_str)
        
        mask = (dataframe[time_column_name] >= start_time) & (dataframe[time_column_name] <= end_time)
        period_df = dataframe[mask].copy()
        
        if not period_df.empty:
            period_df['label'] = label
            labeled_dfs.append(period_df)
            
    if labeled_dfs:
        return pd.concat(labeled_dfs).sort_values(by=time_column_name)
    else:
        return pd.DataFrame()

# --- 執行分割與處理 ---

# 0. 檢查並建立輸出資料夾
if output_directory and not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print(f"\n已建立新資料夾：{output_directory}")

# 1. 產生一個不包含特殊時段的「主要時段列表」
main_periods = [
    p for p in time_periods_with_labels if p not in special_periods_to_extract
]

# 2. 處理並儲存主要資料
main_final_df = process_periods(df, main_periods)
if not main_final_df.empty:
    # 組合完整的儲存路徑與檔名
    output_path_main = os.path.join(output_directory, 'source_data.csv')
    main_final_df.to_csv(output_path_main, index=False, encoding='utf-8-sig')
    print(f"\n主要資料已成功儲存至 {output_path_main}")
    print(f"共包含 {len(main_final_df)} 筆資料。")
else:
    print("\n沒有找到任何主要資料可供儲存。")

# 3. 處理並儲存特殊資料
special_final_df = process_periods(df, special_periods_to_extract)
if not special_final_df.empty:
    # 組合完整的儲存路徑與檔名
    output_path_special = os.path.join(output_directory, 'target_data.csv')
    special_final_df.to_csv(output_path_special, index=False, encoding='utf-8-sig')
    print(f"\n您指定的特殊資料已成功分割並儲存至 {output_path_special}")
    print(f"共包含 {len(special_final_df)} 筆資料。")
else:
    print("\n沒有找到您指定的特殊資料可供分割。")

