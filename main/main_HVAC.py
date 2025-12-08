import os
import subprocess
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from matplotlib.patches import Patch
import time 
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import shutil 

if __name__ == '__main__':
    # 獲取 main_HVAC.py 所在的目錄
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 獲取專案根目錄
    project_root = os.path.dirname(current_dir)
    
    # 1. 設定您的 HVAC 數據集路徑
    dataset_path = os.path.join(project_root, 'datasets', 'HVAC')
    
    if not os.path.exists(dataset_path):
        print(f"錯誤：找不到數據集路徑 {dataset_path}")
        sys.exit()

    # 3. 獲取相關 script 的絕對路徑
    train_script = os.path.join(current_dir, 'train.py')
    eval_script = os.path.join(current_dir, 'eval.py')
    plot_script = os.path.join(current_dir, 'plot.py')
    
    python_executable = sys.executable
    
    # *** 設定特徵 ***
    target_features = [
    # 冷凝盤管阻塞 (Condenser Coil Fault)
    # 'cond_current_1','hp_comp_1','comp_current_1','outdoor_temp', 'return_air_temp' 

    # 蒸發盤管阻塞 (Evaporator Coil Fault)
    # 'fan_current_1','lp_comp_1','comp_current_1','outdoor_temp', 'return_air_temp'

    # 冷媒洩漏 (Refrigerant Leak Fault)
    'lp_comp_1','superheat_1','comp_current_1','return_air_temp','outdoor_temp'   

    # 壓縮機故障 (Compressor Fault)
    # 'lp_comp_1','hp_comp_1','comp_current_1','outdoor_temp', 'return_air_temp'   

    # 冷凝風扇故障
    # 'cond_current_1','outdoor_temp', 'return_air_temp'   

    # 蒸發風扇故障
    # 'fan_current_1','outdoor_temp', 'return_air_temp'   

    #加熱器
    # 'heater_temp', 'outdoor_temp', 'return_air_temp'
    ]
    
    # 4. 明確指定 src 和 trg
    src = "source_data"
    trg = "target_data"
    
    # 2. 定義結果資料夾的變數
    experiments_main_folder = 'results'
    experiment_folder = 'HVAC'

    print(f'======= 正在執行: src: {src} / target: {trg} =======')

    # 5. 定義訓練命令
    command_train = [
        python_executable, train_script,
        '--algo_name', 'dacad',
        '--experiment_folder', experiment_folder,
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
        '--experiments_main_folder', experiments_main_folder,
        '--experiment_folder', experiment_folder,
        '--id_src', src,
        '--id_trg', trg
    ]
    
    # 執行評估
    print("--- 2. 正在執行評估 (eval.py) ---")
    subprocess.run(command_eval, cwd=project_root)

    # 7. 定義繪圖命令 (原有的 plot.py)
    command_plot = [
        python_executable, plot_script,
        '--experiments_main_folder', experiments_main_folder,
        '--experiment_folder', experiment_folder,
        '--id_src', src,
        '--id_trg', trg
    ]
    
    # 執行繪圖
    print("--- 3. 正在執行繪圖 (plot.py) ---")
    subprocess.run(command_plot, cwd=project_root)

    # 8. 複製 'Ours_msltest_' 檔案
    print(f"--- 4. 正在複製 'Ours_msltest_{src}.csv' 檔案 ---")
    
    results_dir = os.path.join(project_root, experiments_main_folder, experiment_folder, f'{src}-{trg}')
    
    try:
        source_filename = f'Ours_msltest_{src}.csv'
        source_file_path = os.path.join(project_root, source_filename)
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy2(source_file_path, results_dir)
        print(f"成功複製 '{source_filename}' 到 {results_dir}")
        
    except FileNotFoundError:
        print(f"錯誤：找不到來源檔案 {source_file_path}")
    except Exception as e:
        print(f"複製檔案時發生錯誤: {e}")

    # ==========================================
    # 9. --- 5. 新增：繪製 y_score 背景標籤圖 (含區段標示) ---
    # ==========================================
    print("--- 5. 正在繪製 y_score 背景標籤圖 (含區段標示) ---")
    
    try:
        # 定義讀取的 csv 路徑
        predictions_csv = os.path.join(results_dir, 'predictions_test_source.csv')
        
        if os.path.exists(predictions_csv):
            df = pd.read_csv(predictions_csv)
            y_score = df['y_pred']
            y_true = df['y']

            # 設定中文字型 (Windows 常用 Microsoft JhengHei, Mac 常用 Arial Unicode MS, Linux 視情況而定)
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False # 解決負號顯示問題

            plt.figure(figsize=(16, 8)) # 加大高度以容納標籤
            plt.plot(y_score, label='y_score (y_pred)', color='blue', linewidth=1)

            # --- 1. 背景顏色填充邏輯 (根據真實標籤 y) ---
            y_array = y_true.values
            n = len(y_array)
            if n > 0:
                start_idx = 0
                current_val = y_array[0]
                for i in range(1, n):
                    if y_array[i] != current_val:
                        color = 'red' if current_val == 1 else 'green'
                        plt.axvspan(start_idx, i, facecolor=color, alpha=0.2)
                        start_idx = i
                        current_val = y_array[i]
                color = 'red' if current_val == 1 else 'green'
                plt.axvspan(start_idx, n-1, facecolor=color, alpha=0.2)

            # --- 2. 標記資料區段 (根據您提供的時間與順序) ---
            # 根據時間排序後的區段資訊 (假設資料頻率為 1Hz，即每秒一點)
            # 順序: 09:46(冷凝) -> 11:01(蒸發盤管) -> 14:19(蒸發風扇) -> 14日(正常)
            
            # 定義各區段長度 (秒數 = 點數)
            # 09:46:00 - 10:16:00 = 30分 = 1800秒 (+1點包含頭尾 = 1801點)
            # 11:01:00 - 11:31:00 = 30分 = 1800秒 (+1點 = 1801點)
            # 14:19:00 - 14:49:00 = 30分 = 1800秒 (+1點 = 1801點)
            # 14日 09:00 - 10:00 = 60分 = 3600秒 (+1點 = 3601點)

            segments = [
                {'name': '冷凝盤管阻塞20%', 'len': 1801},
                {'name': '蒸發盤管阻塞23%', 'len': 1801},
                {'name': '蒸發風扇電流90%', 'len': 1801},
                {'name': '正常資料低溫24度', 'len': 3601},
                {'name': '冷媒洩漏10%', 'len': 3601},
            ]

            current_idx = 0
            y_min, y_max = plt.ylim()
            # 設定文字高度在圖表上方邊緣
            text_y_pos = y_max + (y_max - y_min) * 0.05 

            for seg in segments:
                start = current_idx
                end = current_idx + seg['len'] - 1 # 減1是因為索引從0開始
                
                # 確保不超出實際預測資料長度
                if start >= n:
                    break
                draw_end = min(end, n - 1)

                # 1. 繪製區隔線 (虛線)
                if draw_end < n - 1:
                    plt.axvline(x=draw_end, color='black', linestyle='--', alpha=0.7)

                # 2. 標示文字 (置中)
                mid_point = (start + draw_end) / 2
                plt.text(mid_point, text_y_pos, seg['name'], 
                        ha='center', va='bottom', fontsize=12, fontweight='bold', color='black',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')) # 加個白底讓文字更清楚

                current_idx += seg['len']

            plt.title('Predictions Test Source: y_score with Background & Segment Labels')
            plt.xlabel('Index (Time)')
            plt.ylabel('Score')
            plt.legend(loc='upper right')
            plt.tight_layout()
            
            output_png_name = 'predictions_test_source_marked.png'
            output_png_path = os.path.join(results_dir, output_png_name)
            plt.savefig(output_png_path)
            plt.close()
            
            print(f"標記圖表已儲存至: {output_png_path}")
        else:
            print(f"找不到檔案: {predictions_csv}，無法繪製圖表。")

    except Exception as e:
        print(f"繪製圖表時發生錯誤: {e}")

    except Exception as e:
        print(f"繪製自訂圖表時發生錯誤: {e}")

    print(f"======= HVAC 實驗全部完成 (src: {src}, trg: {trg}) =======")