import pandas as pd
import os # 匯入 os 模組來處理檔案路徑

# (已移除所有內部切割邏輯)

# --- 1. CSV 檔案名稱 ---
# (已更新為您提供的路徑)
csv_file_name = 'D:/DACAD/駕駛室資料/data_org.csv'

# --- 2. 時間欄位名稱 ---
time_column_name = 'DateTime'

# --- 3. 請定義您的資料集 ---
# (*** 這是新的結構 ***)
# 請將您所有的時間段，根據您的需求，分別放入以下三個列表中
# 每個時間段 (包含 '開始', '結束', '標籤') 會被「完整」地分配到一個檔案中
#
# 我已將您所有的時間段都先放在 'source_time_periods' 中，
# 請您手動將它們分配 (剪下並貼上) 到 'target_time_periods' 和 'test_time_periods' 列表中。

source_time_periods = [
    ('2025-04-11 09:14:00', '2025-04-11 09:44:00', 1),#冷凝盤管阻塞20%
    ('2025-04-11 09:46:00', '2025-04-11 10:16:00', 1),#冷凝盤管阻塞30%
    # ('2025-04-11 10:29:00', '2025-04-11 10:59:00', 0),#蒸發盤管阻塞10%
    ('2025-04-11 11:01:00', '2025-04-11 11:31:00', 0),#蒸發盤管阻塞20%
    # ('2025-04-11 11:33:00', '2025-04-11 12:03:00', 0),#蒸發盤管阻塞23%
    ('2025-04-11 12:05:00', '2025-04-11 13:35:00', 1),#正常資料常溫34度
    # ('2025-04-11 13:45:00', '2025-04-11 14:15:00', 1),#蒸發風扇電流90%
    # ('2025-04-11 14:19:00', '2025-04-11 14:49:00', 1),#蒸發風扇電流80%
    # ('2025-04-11 14:52:00', '2025-04-11 15:22:00', 1),#蒸發風扇電流70%
    # ('2025-04-11 15:46:00', '2025-04-11 16:16:00', 1),#正常資料高溫42.7度
    # ('2025-04-14 09:00:00', '2025-04-14 10:00:00', 1),#正常資料低溫24度
    # ('2025-04-14 10:49:00', '2025-04-14 11:45:00', 0),#加熱器運轉
    ('2025-04-14 13:25:00', '2025-04-14 14:25:00', 1),#冷媒洩漏10%
    ('2025-04-14 14:50:00', '2025-04-14 15:50:00', 1),#冷媒洩漏20%
]

target_time_periods = [
    ('2025-04-11 10:29:00', '2025-04-11 10:59:00', 0),#蒸發盤管阻塞10%
    ('2025-04-11 13:45:00', '2025-04-11 14:15:00', 1),#蒸發風扇電流90%
    ('2025-04-11 14:19:00', '2025-04-11 14:49:00', 1),#蒸發風扇電流80%
    ('2025-04-11 14:52:00', '2025-04-11 15:22:00', 1),#蒸發風扇電流70%
]

test_time_periods = [
    ('2025-04-11 11:33:00', '2025-04-11 12:03:00', 0),#蒸發盤管阻塞23%
    ('2025-04-11 15:46:00', '2025-04-11 16:16:00', 1),#正常資料高溫42.7度
    ('2025-04-14 09:00:00', '2025-04-14 10:00:00', 1),#正常資料低溫24度
]


# --- 4. (舊的 'special_periods_to_extract' 已移除) ---

# --- 5. 請修改這裡：設定輸出的資料夾路徑 ---
# (已更新為您提供的路徑)
output_directory = 'D:/DACAD/datasets/HVAC/' # <-- 請在這裡填寫您想儲存的完整路徑

# --- 6. (RANDOM_SEED 已移除) ---


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


# --- 建立一個輔助函式來處理資料篩選與標記 (不儲存) ---
def process_periods_to_df(dataframe, periods_list, dataset_name):
    """
    根據給定的時間段列表，從 dataframe 中篩選、標記並合併資料，
    然後回傳一個合併後的 DataFrame。
    """
    if not periods_list:
        print(f"\n{dataset_name} 列表為空，將產生一個空的 DataFrame。")
        return pd.DataFrame()

    labeled_dfs = []
    
    # 迭代您設定的每一個時間段
    for start_str, end_str, label in periods_list:
        start_time = pd.to_datetime(start_str)
        end_time = pd.to_datetime(end_str)
        
        # 篩選「整個」時間段的資料
        mask = (dataframe[time_column_name] >= start_time) & (dataframe[time_column_name] <= end_time)
        period_df = dataframe[mask].copy()
        
        if not period_df.empty:
            period_df['label'] = label
            labeled_dfs.append(period_df)
        
    if not labeled_dfs:
        print(f"\n在 {dataset_name} 的時間段中沒有找到任何資料。")
        return pd.DataFrame()

    # 合併所有片段並排序
    final_df = pd.concat(labeled_dfs).sort_values(by=time_column_name)
    return final_df

# --- 建立一個輔助函式來儲存檔案 ---
def save_df(df, filename, output_dir):
    """安全地儲存 DataFrame 並印出訊息"""
    if df.empty:
        print(f"\n{filename} 沒有資料，已跳過儲存。")
        return
        
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n檔案已成功儲存至 {output_path} (共 {len(df)} 筆資料)")


# --- 1. 處理 Source Data 並分割為 80/20 ---
print("\n步驟 1: 正在處理 Source Data...")
final_source_df = process_periods_to_df(df, source_time_periods, "Source Data")

if not final_source_df.empty:
    # 按順序計算 80% 的分割點
    split_index = int(len(final_source_df) * 0.8)
    
    # 分割資料
    source_data_df = final_source_df.iloc[:split_index]
    source_val_df = final_source_df.iloc[split_index:]
    
    # 儲存檔案
    save_df(source_data_df, 'source_data_train.csv', output_directory)
    save_df(source_val_df, 'source_data_val.csv', output_directory)
else:
    print("\nSource Data 為空，已跳過 source_data.csv 和 source_data_val.csv 的儲存。")

# --- 2. 處理 Target Data 並分割為 80/20 ---
print("\n步驟 2: 正在處理 Target Data...")
final_target_df = process_periods_to_df(df, target_time_periods, "Target Data")

if not final_target_df.empty:
    # 按順序計算 80% 的分割點
    split_index_target = int(len(final_target_df) * 0.8)
    
    # 分割資料
    target_data_df = final_target_df.iloc[:split_index_target]
    target_val_df = final_target_df.iloc[split_index_target:]
    
    # 儲存檔案
    save_df(target_data_df, 'target_data_train.csv', output_directory)
    save_df(target_val_df, 'target_data_val.csv', output_directory)
else:
    print("\nTarget Data 為空，已跳過 target_data.csv 和 target_data_val.csv 的儲存。")

# --- 3. 處理並儲存 Test Data (不分割) ---
print("\n步驟 3: 正在處理並儲存 Test Data...")
final_test_df = process_periods_to_df(df, test_time_periods, "Test Data")
save_df(final_test_df, 'test_data.csv', output_directory)


print("\n--- 所有資料處理與分割已完成 ---")

