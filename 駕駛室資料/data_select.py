import pandas as pd
import os # 匯入 os 模組來處理檔案路徑

# --- 1. CSV 檔案名稱 ---
# (已更新為您提供的路徑)
csv_file_name = 'D:/DACAD/駕駛室資料/data_org.csv'

# --- 2. 時間欄位名稱 ---
time_column_name = 'DateTime'

# --- 3. 主要的時間段與標籤列表 ---
# 這裡包含您所有的時間定義
# 程式會將這裡的每一個時間段都切成 前40%, 中間20%, 後40%
time_periods_with_labels = [
    ('2025-04-11 09:14:00', '2025-04-11 09:44:00', 0),#冷凝盤管阻塞20%
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
    # ('2025-04-14 10:49:00', '2025-04-14 11:45:00', 0),#加熱器運轉
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
output_directory = 'D:/DACAD/datasets/HVAC/' # <-- 請在這裡填寫您想儲存的完整路徑


# --- 主程式開始 (以下部分通常不需要修改) ---

# 讀取 CSV 檔案
try:
    df = pd.read_csv(csv_file_name)
except FileNotFoundError:
    print(f"錯誤：找不到名為 '{csv_file_name}' 的檔案。")
    print(f"路徑 '{csv_file_name}' 不正確，或檔案不存在。")
    exit()
except Exception as e:
    print(f"讀取 CSV 檔案時發生錯誤: {e}")
    exit()

# 檢查時間欄位是否存在
if time_column_name not in df.columns:
    print(f"錯誤：在您的 CSV 檔案中找不到名為 '{time_column_name}' 的欄位。")
    print(f"檔案中包含的欄位有：{list(df.columns)}")
    exit()

# 將時間欄位轉換為 pandas 的 datetime 格式
df[time_column_name] = pd.to_datetime(df[time_column_name], errors='coerce')
df.dropna(subset=[time_column_name], inplace=True)


# --- 執行分割與處理 ---

# 0. 檢查並建立輸出資料夾
if output_directory and not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print(f"\n已建立新資料夾：{output_directory}")

# 建立三個空列表，分別存放前40%、中間20%、後40%的資料
front_40_dfs = []
middle_20_dfs = []
back_40_dfs = []

# 1. 迭代所有時間段，將它們切成三份
print("\n正在處理時間段並分割資料 (40% - 20% - 40%)...")
for start_str, end_str, label in time_periods_with_labels:
    start_time = pd.to_datetime(start_str)
    end_time = pd.to_datetime(end_str)
    
    # 計算總時長
    duration = end_time - start_time
    
    # 計算兩個分割點
    # 分割點1: 前 40% 結束
    split_point_1 = start_time + (duration * 0.4)
    # 分割點2: 中間 20% 結束 (即總時長的 60%)
    split_point_2 = start_time + (duration * 0.6)
    
    # 篩選前 40% (>= start, < split_point_1)
    mask_front = (df[time_column_name] >= start_time) & (df[time_column_name] < split_point_1)
    front_df = df[mask_front].copy()
    
    # 篩選中間 20% (>= split_point_1, < split_point_2)
    # 注意：您的原始資料 '2026-01-01 04:00:00' 到 '04:30:00' 這種30分鐘的區間，
    # 中間20% (6分鐘) 可能沒有資料點，這是正常的。
    mask_middle = (df[time_column_name] >= split_point_1) & (df[time_column_name] < split_point_2)
    middle_df = df[mask_middle].copy()

    # 篩選後 40% (>= split_point_2, <= end)
    mask_back = (df[time_column_name] >= split_point_2) & (df[time_column_name] <= end_time)
    back_df = df[mask_back].copy()
    
    # 標記並儲存前半段
    if not front_df.empty:
        front_df['label'] = label
        front_40_dfs.append(front_df)
        
    # 標記並儲存中間段
    if not middle_df.empty:
        middle_df['label'] = label
        middle_20_dfs.append(middle_df)

    # 標記並儲存後半段
    if not back_df.empty:
        back_df['label'] = label
        back_40_dfs.append(back_df)

print("資料分割完成。")

# 2. 處理並儲存「前 40%」組合資料
if front_40_dfs:
    front_final_df = pd.concat(front_40_dfs).sort_values(by=time_column_name)
    # 組合完整的儲存路徑與檔名
    output_path_front = os.path.join(output_directory, 'source_data.csv')
    front_final_df.to_csv(output_path_front, index=False, encoding='utf-8-sig')
    print(f"\n所有時段的「前 40%」資料已成功儲存至 {output_path_front}")
    print(f"共包含 {len(front_final_df)} 筆資料。")
else:
    print("\n沒有找到任何「前 40%」資料可供儲存。")

# 3. 處理並儲存「中間 20%」組合資料
if middle_20_dfs:
    middle_final_df = pd.concat(middle_20_dfs).sort_values(by=time_column_name)
    # 組合完整的儲存路徑與檔名
    output_path_middle = os.path.join(output_directory, 'test.csv')
    middle_final_df.to_csv(output_path_middle, index=False, encoding='utf-8-sig')
    print(f"\n所有時段的「中間 20%」資料已成功儲存至 {output_path_middle}")
    print(f"共包含 {len(middle_final_df)} 筆資料。")
else:
    print("\n沒有找到任何「中間 20%」資料可供儲存。")

# 4. 處理並儲存「後 40%」組合資料
if back_40_dfs:
    back_final_df = pd.concat(back_40_dfs).sort_values(by=time_column_name)
    # 組合完整的儲存路徑與檔名
    output_path_back = os.path.join(output_directory, 'target_data.csv')
    back_final_df.to_csv(output_path_back, index=False, encoding='utf-8-sig')
    print(f"\n所有時段的「後 40%」資料已成功儲存至 {output_path_back}")
    print(f"共包含 {len(back_final_df)} 筆資料。")
else:
    print("\n沒有找到任何「後 40%」資料可供儲存。")

