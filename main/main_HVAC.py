# 檔案: main/main_HVAC.py (已整合自動繪圖功能)

import os
import subprocess
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from matplotlib.patches import Patch
import time # 用於處理檔案延遲
from datetime import datetime, timedelta

# =============================================================================
# START: 新增的輔助函式 (來自 plot_timeline.py)
# =============================================================================

def find_best_f1_threshold(scores, true_labels):
    # 此處是您的閾值尋找邏輯，保持不變
    # ... (原有代碼)
    thresholds = np.sort(np.unique(scores))
    best_f1 = -1
    best_threshold = 0

    for th in thresholds:
        pred_labels = (scores > th).astype(int)
        tp = np.sum((pred_labels == 1) & (true_labels == 1))
        fp = np.sum((pred_labels == 1) & (true_labels == 0))
        fn = np.sum((pred_labels == 0) & (true_labels == 1))

        if tp + fp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = th
    return best_threshold


def plot_anomaly_timeline(prediction_csv_path, output_dir, dataset_name):
    print(f"--- 正在繪製: {prediction_csv_path.split('/')[-1]} ---")

    pred_df = pd.read_csv(prediction_csv_path)

    # 檢查必要的欄位
    if 'y_pred' not in pred_df.columns or 'y' not in pred_df.columns or 'stay_hour' not in pred_df.columns:
        print(f"  [繪圖錯誤]：預測 CSV 必須包含 'y_pred', 'y', 和 'stay_hour' 欄位。跳過此圖表。")
        return

    scores = pred_df['y_pred'].values
    true_labels = pred_df['y'].values
    stay_hours = pred_df['stay_hour'].values

    # --- 1. 建立時間軸 (X 軸) ---
    # 假設 'stay_hour' 是從 0 開始的小時數。
    # 我們需要一個參考日期來建立 datetime 物件。
    # 這裡使用一個假的起始日期 2025-04-11 00:00:00，你可以根據需要調整
    start_time = datetime(2025, 4, 11, 0, 0, 0)
    # 根據 stay_hour 創建 datetime 物件，可以假設 stay_hour 是分鐘數，或者就是小時數
    # 這裡假設 stay_hour 是一個連續的、代表「分鐘」的索引
    # 如果 stay_hour 是小時，則調整 timedelta(minutes=h) 為 timedelta(hours=h)
    time_index = [start_time + timedelta(minutes=int(h)) for h in stay_hours] # 假設 stay_hour 是分鐘數

    # --- 2. 找到最佳門檻值並產生 0/1 預測 (此處用於繪製模型的預測高亮區，如果需要) ---
    # 這裡我們只繪製真實標籤的背景高亮，所以模型的預測高亮可以暫時不計算
    # 如果您想也顯示模型的預測高亮，可以取消註解以下兩行
    # best_threshold = find_best_f1_threshold(scores, true_labels)
    # model_predictions_binary = (scores > best_threshold).astype(int)

    # --- 3. 繪圖 ---
    plt.figure(figsize=(20, 5)) # 調整圖表大小
    plt.plot(time_index, scores, label='Anomaly Score (NLL)', color='red', linewidth=2) # 使用 time_index

    # 繪製真實標籤的背景高亮
    # 遍歷真實標籤，尋找異常區間 (y=1)
    anomaly_regions = []
    in_anomaly = False
    start_anomaly_idx = 0

    for i in range(len(true_labels)):
        if true_labels[i] == 1 and not in_anomaly:
            in_anomaly = True
            start_anomaly_idx = i
        elif true_labels[i] == 0 and in_anomaly:
            in_anomaly = False
            anomaly_regions.append((time_index[start_anomaly_idx], time_index[i-1])) # 結束時間是前一個點
    # 如果數據結束時仍在異常狀態
    if in_anomaly:
        anomaly_regions.append((time_index[start_anomaly_idx], time_index[-1]))

    # 繪製高亮區
    for start, end in anomaly_regions:
        plt.axvspan(start, end, color='yellow', alpha=0.3, label='True Anomaly' if 'True Anomaly' not in [l.get_label() for l in plt.gca().lines + plt.gca().patches] else "")

    # 繪製正常區 (淺綠色)
    normal_regions = []
    in_normal = False
    start_normal_idx = 0

    for i in range(len(true_labels)):
        if true_labels[i] == 0 and not in_normal:
            in_normal = True
            start_normal_idx = i
        elif true_labels[i] == 1 and in_normal:
            in_normal = False
            normal_regions.append((time_index[start_normal_idx], time_index[i-1]))
    if in_normal:
        normal_regions.append((time_index[start_normal_idx], time_index[-1]))

    for start, end in normal_regions:
        plt.axvspan(start, end, color='lightgreen', alpha=0.3, label='Normal' if 'Normal' not in [l.get_label() for l in plt.gca().lines + plt.gca().patches] else "")


    # 圖表美化
    plt.title('Anomaly Score Over Time with Background Highlight', fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.grid(True)
    plt.legend(loc='upper right') # 將圖例放在右上角
    plt.xticks(rotation=30, ha='right') # 旋轉 x 軸標籤，使其不重疊
    plt.tight_layout() # 自動調整佈局，防止標籤重疊

    # --- 儲存圖片 (已修正路徑問題) ---
    
    # 1. 使用 os.path.basename 安全地取得檔案名稱 (例如 "predictions_test_source.csv")
    base_filename = os.path.basename(prediction_csv_path)
    
    # 2. 移除 .csv 副檔名
    filename_without_ext = base_filename.replace('.csv', '')
    
    # 3. 組合新的輸出檔名
    output_filename = f"anomaly_timeline_{dataset_name}_{filename_without_ext}.png"
    
    # 4. 使用 os.path.join 安全地組合輸出路徑 (避免 / 和 \ 的混淆)
    full_output_path = os.path.join(output_dir, output_filename)
    
    # 5. 儲存
    try:
        plt.savefig(full_output_path)
        print(f"  [繪圖成功] 圖表已儲存至: {full_output_path}")
    except Exception as e:
        print(f"  [繪圖錯誤] 儲存圖片失敗: {e}")
        
    plt.close() # 關閉圖表，釋放記憶體

    # (原有的 print 訊息可以移除，因為上面已經印出儲存路徑)
    # print(f"======= 繪圖完成: {output_filename} =======")
# =============================================================================
# END: 新增的輔助函式
# =============================================================================


if __name__ == '__main__':
    # 獲取 main_HVAC.py 所在的目錄 (例如 d:\DACAD\main)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 獲取專案根目錄 (例如 d:\DACAD)
    project_root = os.path.dirname(current_dir)
    
    # 1. 設定您的 HVAC 數據集路徑
    dataset_path = os.path.join(project_root, 'datasets', 'HVAC')
    
    if not os.path.exists(dataset_path):
        print(f"錯誤：找不到數據集路徑 {dataset_path}")
        sys.exit()
        
    all_files = os.listdir(dataset_path)

    # 2. 找到所有 .csv 檔案
    files = [name[:-4] for name in all_files if name.endswith('.csv')]
    files = sorted(files)
    
    # 3. 獲取 train.py 和 eval.py 的絕對路徑
    train_script = os.path.join(current_dir, 'train.py')
    eval_script = os.path.join(current_dir, 'eval.py')
    
    python_executable = sys.executable
    
    # *** 請將 'Your_Feature_Name_1' 替換為您 CSV 中的實際欄位名稱 ***
    target_features = [
        'hp_comp_1', 
        'lp_comp_1', 
        'comp_current_1', 
        'cond_current_1', 
        'return_air_temp', 
        'superheat_1', 
        'lp_plate_temp_1'
    ]
    
    # 4. 循環遍歷所有 來源(src) -> 目標(trg) 組合
    for src in files:
        for trg in files:
            if src != trg:
                print(f'======= 正在執行: src: {src} / target: {trg} =======')

                # 5. 定義訓練命令
                command_train = [
                    python_executable, train_script,
                    '--algo_name', 'dacad',
                    '--experiment_folder', 'HVAC',           
                    '--path_src', dataset_path,
                    '--path_trg', dataset_path,
                    '--id_src', src,
                    '--id_trg', trg,
                    '--num_epochs', '5',
                    '--batch_size', '128',
                    '--eval_batch_size', '256',
                    '--learning_rate', '1e-4',
                    '--dropout', '0.1',
                    '--weight_decay', '1e-4',
                    '--num_channels_TCN', '128-256-512',
                    '--dilation_factor_TCN', '3',
                    '--kernel_size_TCN', '7',
                    '--hidden_dim_MLP', '1024',
                    '--queue_size', '98304',
                    '--momentum', '0.99',
                    '--features', *target_features 
                ]
                
                # 在專案根目錄 (d:\DACAD) 中執行
                subprocess.run(command_train, cwd=project_root)

                # 6. 定義評估命令
                command_eval = [
                    python_executable, eval_script,
                    '--experiments_main_folder', 'results',
                    '--experiment_folder', 'HVAC',
                    '--id_src', src,
                    '--id_trg', trg
                ]
                
                # 在專案根目錄 (d:\DACAD) 中執行
                subprocess.run(command_eval, cwd=project_root)

                # ===================================================================
                # START: 新增自動繪圖區塊
                # ===================================================================
                print(f'\n======= 正在為 {src}-{trg} 繪製結果圖表 =======')
                
                # 1. 定義實驗結果路徑
                current_experiment_dir = os.path.join(project_root, 'results', 'HVAC', f'{src}-{trg}')
                
                # 2. 繪製 Source 測試集結果
                # eval.py 會儲存 "predictions_test_source.csv"
                try:
                    # 組合 CSV 完整路徑
                    source_csv_path = os.path.join(current_experiment_dir, "predictions_test_source.csv")
                    # *** 修改：傳遞 3 個參數 ***
                    # 1. 完整 CSV 路徑, 2. 儲存圖片的目錄, 3. 用於命名的字串
                    plot_anomaly_timeline(source_csv_path, current_experiment_dir, f'{src}-{trg}')
                except Exception as e:
                    print(f"  [繪圖錯誤] 繪製 predictions_test_source.csv 時發生未預期錯誤: {e}")

                # 3. 繪製 Target 測試集結果
                # eval.py 會儲存 "predictions_test_target.csv"
                try:
                    # 組合 CSV 完整路徑
                    target_csv_path = os.path.join(current_experiment_dir, "predictions_test_target.csv")
                    # *** 修改：傳遞 3 個參數 ***
                    plot_anomaly_timeline(target_csv_path, current_experiment_dir, f'{src}-{trg}')
                except Exception as e:
                    print(f"  [繪圖錯誤] 繪製 predictions_test_target.csv 時發生未預期錯誤: {e}")
                
                print(f'======= 繪圖完成: {src}-{trg} =======\n')
                # ===================================================================
                # END: 新增自動繪圖區塊
                # ===================================================================

    print("======= 所有 HVAC 實驗已完成 =======")