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
        
    files = ["source_data", "target_data"]
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
                    '--num_epochs', '10',
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

    print("======= 所有 HVAC 實驗已完成 =======")