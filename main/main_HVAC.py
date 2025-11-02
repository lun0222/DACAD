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
import matplotlib.dates as mdates
import shutil # --- 1. 新增：導入 shutil 模組 ---

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

    # 3. 獲取 train.py, eval.py 和 plot.py 的絕對路徑
    train_script = os.path.join(current_dir, 'train.py')
    eval_script = os.path.join(current_dir, 'eval.py')
    plot_script = os.path.join(current_dir, 'plot.py')
    
    python_executable = sys.executable
    
    # *** 請將 'Your_Feature_Name_1' 替換為您 CSV 中的實際欄位名稱 ***
    target_features = [

    # 冷凝盤管阻塞 (Condenser Coil Fault)
    # 'cond_current_1','hp_comp_1','comp_current_1','outdoor_temp', 'return_air_temp' 
    # 'lp_comp_1','superheat_1','comp_current_1','outdoor_temp','return_air_temp'   

    # 蒸發盤管阻塞 (Evaporator Coil Fault)
    # 'fan_current_1','lp_comp_1','comp_current_1','outdoor_temp', 'return_air_temp'

    # 冷媒洩漏 (Refrigerant Leak Fault)
    'lp_comp_1','superheat_1','comp_current_1','return_air_temp'   

    # 壓縮機故障 (Compressor Fault)
    #'cond_current_1','hp_comp_1','comp_current_1','outdoor_temp', 'return_air_temp']   

    # 冷凝風扇故障
    #'cond_current_1','outdoor_temp', 'return_air_temp']   

    # 蒸發風扇故障
    #'fan_current_1','outdoor_temp', 'return_air_temp']   

    #加熱器
    #'heater_temp', 'outdoor_temp', 'return_air_temp'

    ]
    
    # 4. 明確指定 src 和 trg (只執行 source_data -> target_data)
    src = "source_data"
    trg = "target_data"
    
    # --- 2. 新增：定義結果資料夾的變數 ---
    experiments_main_folder = 'results'
    experiment_folder = 'HVAC'

    print(f'======= 正在執行: src: {src} / target: {trg} =======')

    # 5. 定義訓練命令
    command_train = [
        python_executable, train_script,
        '--algo_name', 'dacad',
        '--experiment_folder', experiment_folder, # 使用變數
        '--path_src', dataset_path,
        '--path_trg', dataset_path,
        '--id_src', src,
        '--id_trg', trg,
        '--num_epochs', '50',
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
    
    # 執行訓練
    print("--- 1. 正在執行訓練 (train.py) ---")
    subprocess.run(command_train, cwd=project_root)

    # 6. 定義評估命令
    command_eval = [
        python_executable, eval_script,
        '--experiments_main_folder', experiments_main_folder, # 使用變數
        '--experiment_folder', experiment_folder, # 使用變數
        '--id_src', src,
        '--id_trg', trg
    ]
    
    # 執行評估
    print("--- 2. 正在執行評估 (eval.py) ---")
    subprocess.run(command_eval, cwd=project_root)

    # 7. 定義繪圖命令
    command_plot = [
        python_executable, plot_script,
        '--experiments_main_folder', experiments_main_folder, # 使用變數
        '--experiment_folder', experiment_folder, # 使用變數
        '--id_src', src,
        '--id_trg', trg
    ]
    
    # 執行繪圖
    print("--- 3. 正在執行繪圖 (plot.py) ---")
    subprocess.run(command_plot, cwd=project_root)

    # 8. --- 3. 新增：複製 'Ours_msltest_' 檔案 ---
    print(f"--- 4. 正在複製 'Ours_msltest_{src}.csv' 檔案 ---")
    
    try:
        # 建立來源檔案名稱 (例如: Ours_msltest_source_data.csv)
        source_filename = f'Ours_msltest_{src}.csv'
        # 建立來源檔案的完整路徑 (例如: D:\DACAD\Ours_msltest_source_data.csv)
        source_file_path = os.path.join(project_root, source_filename)
        
        # 建立目標資料夾路徑 (例如: D:\DACAD\results\HVAC\source_data-target_data)
        destination_dir = os.path.join(project_root, experiments_main_folder, experiment_folder, f'{src}-{trg}')
        
        # 確保目標資料夾存在 (eval.py 應該已經建立了, 但多做一層保險)
        os.makedirs(destination_dir, exist_ok=True)
        
        # 執行複製 (shutil.copy2 會保留原始檔案的中繼資料)
        shutil.copy2(source_file_path, destination_dir)
        print(f"成功複製 '{source_filename}' 到 {destination_dir}")
        
    except FileNotFoundError:
        print(f"錯誤：找不到來源檔案 {source_file_path}")
    except Exception as e:
        print(f"複製檔案時發生錯誤: {e}")

    print(f"======= HVAC 實驗已完成 (src: {src}, trg: {trg}) =======")