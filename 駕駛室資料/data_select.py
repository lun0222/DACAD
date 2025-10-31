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
    # ('2025-04-11 11:01:00', '2025-04-11 11:31:00', 0),#蒸發盤管阻塞20%
    ('2025-04-11 11:33:00', '2025-04-11 12:03:00', 0),#蒸發盤管阻塞23%
    ('2025-04-11 12:05:00', '2025-04-11 13:35:00', 1),#正常資料常溫34度
    # ('2025-04-11 13:45:00', '2025-04-11 14:15:00', 1),#蒸發風扇電流90%
    # ('2025-04-11 14:19:00', '2025-04-11 14:49:00', 1),#蒸發風扇電流80%
    # ('2025-04-11 14:52:00', '2025-04-11 15:22:00', 1),#蒸發風扇電流70%
    # ('2025-04-11 15:46:00', '2025-04-11 16:16:00', 1),#正常資料高溫42.7度
    ('2025-04-14 09:00:00', '2025-04-14 10:00:00', 1),#正常資料低溫24度
    # ('2025-04-14 10:49:00', '2025-04-14 11:45:00', 0),#加熱器運轉
    # ('2025-04-14 13:25:00', '2025-04-14 14:25:00', 1),#冷媒洩漏10%
    # ('2025-04-14 14:50:00', '2025-04-14 15:50:00', 1),#冷媒洩漏20%
]

target_time_periods = [
    ('2025-04-11 10:29:00', '2025-04-11 10:59:00', 0),#蒸發盤管阻塞10%
    ('2025-04-11 13:45:00', '2025-04-11 14:15:00', 1),#蒸發風扇電流90%
    ('2025-04-11 14:19:00', '2025-04-11 14:49:00', 1),#蒸發風扇電流80%
    ('2025-04-11 14:52:00', '2025-04-11 15:22:00', 1),#蒸發風扇電流70%
    ('2025-04-14 13:25:00', '2025-04-14 14:25:00', 1),#冷媒洩漏10%
    ('2025-04-14 14:50:00', '2025-04-14 15:50:00', 1),#冷媒洩漏20%
]

test_time_periods = [
    ('2025-04-11 11:01:00', '2025-04-11 11:31:00', 0),#蒸發盤管阻塞20%
    ('2025-04-11 15:46:00', '2025-04-11 16:16:00', 1),#正常資料高溫42.7度
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


# --- 建立一個輔助函式來處理資料篩選、標記與儲存 ---
def process_and_save_periods(dataframe, periods_list, filename, output_dir):
    """
    根據給定的時間段列表，從 dataframe 中篩選、標記並合併資料，
    然後儲存成一個檔案。
    """
    if not periods_list:
        print(f"\n列表 {filename} 為空，跳過儲存。")
        return

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
        print(f"\n在 {filename} 的時間段中沒有找到任何資料。")
        return

    # 合併所有片段並儲存
    final_df = pd.concat(labeled_dfs).sort_values(by=time_column_name)
    output_path = os.path.join(output_dir, filename)
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n檔案已成功儲存至 {output_path} (共 {len(final_df)} 筆資料)")


# --- 1. 彙整並儲存 3 個檔案 ---
print("\n步驟 1: 正在處理並儲存 Source Data...")
process_and_save_periods(df, source_time_periods, 'source_data_train.csv', output_directory)

print("\n步驟 2: 正在處理並儲存 Target Data...")
process_and_save_periods(df, target_time_periods, 'target_data_train.csv', output_directory)

print("\n步驟 3: 正在處理並儲存 Test Data...")
process_and_save_periods(df, test_time_periods, 'test_data.csv', output_directory)

print("\n--- 所有資料處理與分割已完成 ---")
