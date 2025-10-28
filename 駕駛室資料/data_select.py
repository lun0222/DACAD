import pandas as pd
import os # 匯入 os 模組來處理檔案路徑

# (不再需要 scikit-learn，因為我們是按順序切割)

# --- 1. CSV 檔案名稱 ---
# (已更新為您提供的路徑)
csv_file_name = 'D:/DACAD/駕駛室資料/data_org.csv'

# --- 2. 時間欄位名稱 ---
time_column_name = 'DateTime'

# --- 3. 主要的時間段與標籤列表 ---
# 這裡包含您所有的時間定義
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
    # ... (其他被註解的時段)
]

# --- 4. (舊的 'special_periods_to_extract' 已移除) ---

# --- 5. 請修改這裡：設定輸出的資料夾路徑 ---
# (已更新為您提供的路徑)
output_directory = 'D:/DACAD/datasets/HVAC/' # <-- 請在這裡填寫您想儲存的完整路徑

# --- 6. (RANDOM_SEED 已移除，因為我們現在按順序分割) ---


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
# 確保原始資料是按時間排序的
df.sort_values(by=time_column_name, inplace=True)


# --- 執行分割與處理 ---

# 0. 檢查並建立輸出資料夾
if output_directory and not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print(f"\n已建立新資料夾：{output_directory}")

# 建立 5 個空列表，分別存放 5 個最終檔案的資料片段
source_train_dfs = []
source_val_dfs = []
target_train_dfs = []
target_val_dfs = []
test_dfs = [] # 統一的測試集

# --- 建立一個輔助函式，用於按順序切割 30/10/30/10/20 ---
def split_period_into_five(period_df, label):
    """將單一時段的 DataFrame 按順序切成 30/10/30/10/20 五份"""
    if period_df.empty:
        return None, None, None, None, None
        
    # 標記標籤
    period_df = period_df.copy()
    period_df['label'] = label
    
    total_rows = len(period_df)
    
    # 計算分割點
    s1 = int(total_rows * 0.3)  # 30%
    s2 = int(total_rows * 0.4)  # 30% + 10% = 40%
    s3 = int(total_rows * 0.7)  # 40% + 30% = 70%
    s4 = int(total_rows * 0.8)  # 70% + 10% = 80%
    # 剩下的 20% (80% -> 100%) 為 s4 之後

    # 按順序切割
    source_train_part = period_df.iloc[:s1]    # 前 30%
    source_val_part = period_df.iloc[s1:s2]    # 中 10% (30% -> 40%)
    target_train_part = period_df.iloc[s2:s3]    # 中 30% (40% -> 70%)
    target_val_part = period_df.iloc[s3:s4]    # 中 10% (70% -> 80%)
    test_part = period_df.iloc[s4:]            # 後 20% (80% -> 100%)
    
    return source_train_part, source_val_part, target_train_part, target_val_part, test_part

# --- 1. 迭代所有時間段，進行 30/10/30/10/20 分割 ---
print("\n步驟 1: 正在處理每個時間段...")
for start_str, end_str, label in time_periods_with_labels:
    start_time = pd.to_datetime(start_str)
    end_time = pd.to_datetime(end_str)
    
    # (50/50 分割邏輯已移除)
    
    # 篩選「整個」時間段的資料
    mask_full_period = (df[time_column_name] >= start_time) & (df[time_column_name] <= end_time)
    full_period_df = df[mask_full_period]

    # --- 處理 Source Data ---
    s_train, s_val, t_train, t_val, test = split_period_into_five(full_period_df, label)
    
    if s_train is not None and not s_train.empty: source_train_dfs.append(s_train)
    if s_val is not None and not s_val.empty:   source_val_dfs.append(s_val)
    if t_train is not None and not t_train.empty: target_train_dfs.append(t_train)
    if t_val is not None and not t_val.empty:   target_val_dfs.append(t_val)
    if test is not None and not test.empty:    test_dfs.append(test)

print("所有時間段都已按 30/10/30/10/20 的順序分割完成。")

# --- 建立一個輔助函式來合併與儲存 ---
def concat_and_save(dfs_list, filename, output_dir):
    """將資料片段列表合併、排序並儲存"""
    if not dfs_list:
        print(f"\n沒有找到資料可儲存為 {filename}。")
        return

    # 合併來自所有時段的片段
    final_df = pd.concat(dfs_list)
    
    # 確保最終檔案是按時間排序的
    final_df.sort_values(by=time_column_name, inplace=True)
    
    # 組合路徑並儲存
    output_path = os.path.join(output_dir, filename)
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n檔案已成功儲存至 {output_path} (共 {len(final_df)} 筆資料)")

# --- 2. 彙整並儲存 5 個檔案 ---
print("\n步驟 2: 正在彙整並儲存 5 個最終檔案...")

# Source 檔案
concat_and_save(source_train_dfs, 'source_train.csv', output_directory)
concat_and_save(source_val_dfs,   'source_val.csv',   output_directory)

# Target 檔案
concat_and_save(target_train_dfs, 'target_train.csv', output_directory)
concat_and_save(target_val_dfs,   'target_val.csv',   output_directory)

# Test 檔案
concat_and_save(test_dfs, 'test.csv', output_directory)

print("\n--- 所有資料處理與分割已完成 ---")

