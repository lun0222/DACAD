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

# =============================================================================
# START: 新增的輔助函式 (來自 plot_timeline.py)
# =============================================================================

def find_best_f1_threshold(scores, labels):
    """
    從 scores (預測分數) 和 labels (真實標籤) 中找到最佳 F1 門檻值。
    """
    if not isinstance(scores, np.ndarray):
        scores = np.array(scores)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    max_f1 = 0.0
    best_th = 0.0

    if len(np.unique(labels)) == 1:
        print("  [繪圖警告]：真實標籤只包含單一類別。F1 門檻值可能無意義。")
        return 0.5 

    thresholds = np.unique(scores)
    
    if len(thresholds) > 1000:
       thresholds = np.linspace(thresholds.min(), thresholds.max(), 1000)

    for th in thresholds:
        pred_labels = (scores >= th).astype(int)
        f1 = f1_score(labels, pred_labels, zero_division=0) # 增加 zero_division=0 避免警告
        
        if f1 > max_f1:
            max_f1 = f1
            best_th = th
            
    print(f"  [繪圖資訊] 找到最佳 F1 門檻值: {best_th:.4f} (對應 F1: {max_f1:.4f})")
    return best_th

def plot_anomaly_timeline(experiment_dir_path, result_file_name):
    """
    主繪圖函式：繪製預測與真實標籤的時間軸
    """
    print(f'--- 正在繪製: {result_file_name} ---')
    
    # --- 1. 定義路徑 ---
    pred_csv_path = os.path.join(experiment_dir_path, result_file_name)
    
    # 根據 dataset.py，窗口大小 (wsz) 固定為 100
    window_size = 100 
    
    # --- 2. 載入預測結果 ---
    # 有時檔案系統寫入會延遲，我們給它一點時間
    max_retries = 5
    for i in range(max_retries):
        if os.path.exists(pred_csv_path):
            break
        print(f"  [繪圖資訊] 等待 {result_file_name} 檔案生成... ({i+1}/{max_retries})")
        time.sleep(1) # 等待 1 秒
        
    if not os.path.exists(pred_csv_path):
        print(f"  [繪圖錯誤]：找不到預測檔案: {pred_csv_path}。跳過此圖表。")
        return
        
    try:
        pred_df = pd.read_csv(pred_csv_path)
    except pd.errors.EmptyDataError:
        print(f"  [繪圖錯誤]：預測檔案 {pred_csv_path} 為空。跳過此圖表。")
        return
    
    if 'prediction' not in pred_df.columns or 'label' not in pred_df.columns:
        print(f"  [繪圖錯誤]：預測 CSV 必須包含 'prediction' 和 'label' 欄位。跳過此圖表。")
        return

    scores = pred_df['prediction'].values
    true_labels = pred_df['label'].values

    # --- 3. 找到最佳門檻值並產生 0/1 預測 ---
    best_threshold = find_best_f1_threshold(scores, true_labels)
    model_predictions = (scores >= best_threshold).astype(int)

    # --- 4. 繪製時間軸 ---
    plt.figure(figsize=(20, 4))
    
    colors_true = {1: '#A93226', 0: '#229954'} # 深紅, 深綠
    colors_pred = {1: '#FADBD8', 0: '#D5F5E3'} # 淺紅, 淺綠

    for i, pred in enumerate(model_predictions):
        color = colors_pred[pred]
        plt.axvspan(i, i + window_size, ymin=0.5, ymax=1.0, color=color, alpha=0.9, linewidth=0)

    for i, label in enumerate(true_labels):
        color = colors_true[label]
        plt.axvspan(i, i + window_size, ymin=0.0, ymax=0.5, color=color, alpha=0.9, linewidth=0)

    # --- 5. 格式化圖表 ---
    plt.title(f'Anomaly Detection Timeline\n(File: {result_file_name})')
    plt.xlabel('Time Step (Window Index)')
    plt.xlim(0, len(true_labels) + window_size)
    plt.ylim(0, 1)
    plt.yticks([0.25, 0.75], ['True Labels (Ground Truth)', 'Model Predictions'])
    
    legend_elements = [
        Patch(facecolor=colors_pred[1], label='Predicted Anomaly'),
        Patch(facecolor=colors_pred[0], label='Predicted Normal'),
        Patch(facecolor=colors_true[1], label='True Anomaly'),
        Patch(facecolor=colors_true[0], label='True Normal')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    
    # 儲存圖表
    output_filename = f"timeline_{result_file_name.replace('.csv', '')}.png"
    output_path = os.path.join(experiment_dir_path, output_filename)
    
    plt.savefig(output_path)
    print(f"  [繪圖成功] 時間軸圖表已儲存至: {output_path}")
    plt.close()

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
                    plot_anomaly_timeline(current_experiment_dir, "predictions_test_source.csv")
                except Exception as e:
                    print(f"  [繪圖錯誤] 繪製 predictions_test_source.csv 時發生未預期錯誤: {e}")

                # 3. 繪製 Target 測試集結果
                # eval.py 會儲存 "predictions_test_target.csv"
                try:
                    plot_anomaly_timeline(current_experiment_dir, "predictions_test_target.csv")
                except Exception as e:
                    print(f"  [繪圖錯誤] 繪製 predictions_test_target.csv 時發生未預期錯誤: {e}")
                
                print(f'======= 繪圖完成: {src}-{trg} =======\n')
                # ===================================================================
                # END: 新增自動繪圖區塊
                # ===================================================================

    print("======= 所有 HVAC 實驗已完成 =======")