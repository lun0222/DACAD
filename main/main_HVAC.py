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
    # 'cond_current_1','hp_comp_1','comp_current_1','return_air_temp' ,'outdoor_temp'

    # 蒸發盤管阻塞 (Evaporator Coil Fault)
    # 'fan_current_1','lp_comp_1','comp_current_1','superheat_1','return_air_temp','outdoor_temp'

    # 冷媒洩漏 (Refrigerant Leak Fault)
    # 'hp_comp_1','lp_comp_1','superheat_1','comp_current_1','return_air_temp','outdoor_temp'

    # 壓縮機故障 (Compressor Fault)
    # 'lp_comp_1','hp_comp_1','comp_current_1','cond_current_1','return_air_temp','outdoor_temp'

    # 冷凝風扇故障
    # 'hp_comp_1','cond_current_1','comp_current_1','return_air_temp','outdoor_temp'

    # 蒸發風扇故障
    # 'fan_current_1','lp_comp_1','comp_current_1','return_air_temp','outdoor_temp'

    #加熱器
    'heater_temp','return_air_temp','outdoor_temp'
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
    # 9. --- 5. 新增：繪製 y_score 背景標籤圖 (閾值判斷版) ---
    # ==========================================
    print("--- 5. 正在繪製 y_score 背景標籤圖 (基於最佳閾值著色) ---")
    
    try:
        from sklearn.metrics import precision_recall_curve
        
        # 定義讀取的 csv 路徑
        predictions_csv = os.path.join(results_dir, 'predictions_test_source.csv')
        
        if os.path.exists(predictions_csv):
            df = pd.read_csv(predictions_csv)
            y_score = df['y_pred']
            y_true = df['y']

            # --- 1. 計算最佳閾值 (Best Threshold) ---
            precision, recall, thresholds = precision_recall_curve(y_true, y_score)
            # 計算每個閾值下的 F1 Score
            numerator = 2 * recall * precision
            denominator = recall + precision
            # 避免除以 0
            f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
            
            # 找出最大 F1 Score 對應的索引
            best_idx = np.argmax(f1_scores)
            
            # 取得最佳閾值 (如果索引超出 thresholds 範圍，取最後一個)
            if best_idx < len(thresholds):
                best_thr = thresholds[best_idx]
            else:
                best_thr = thresholds[-1]
            
            print(f"計算出的最佳閾值 (Best Threshold): {best_thr:.4f}, 最高 F1: {f1_scores[best_idx]:.4f}")

            # --- 2. 開始繪圖 ---
            # 設定中文字型
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False 

            plt.figure(figsize=(16, 8))
            plt.plot(y_score, label='y_score (y_pred)', color='blue', linewidth=1)
            
            # 畫出閾值線 (方便觀察)
            plt.axhline(y=best_thr, color='black', linestyle=':', alpha=0.8, label=f'Threshold: {best_thr:.2f}')

            # --- 3. 背景顏色填充邏輯 (根據 y_pred > best_thr) ---
            # 產生預測標籤陣列: 大於閾值為 1 (異常/紅色), 小於等於為 0 (正常/綠色)
            pred_labels = (y_score > best_thr).astype(int).values
            
            n = len(pred_labels)
            if n > 0:
                start_idx = 0
                current_val = pred_labels[0]
                for i in range(1, n):
                    if pred_labels[i] != current_val:
                        # 顏色邏輯：1 (大於閾值) -> 紅色, 0 -> 綠色
                        color = 'red' if current_val == 1 else 'green'
                        plt.axvspan(start_idx, i, facecolor=color, alpha=0.3) # 您要求的 alpha=0.3
                        start_idx = i
                        current_val = pred_labels[i]
                
                # 繪製最後一段
                color = 'red' if current_val == 1 else 'green'
                plt.axvspan(start_idx, n-1, facecolor=color, alpha=0.3)

            # --- 4. 標記資料區段 (保持您原本的區段設定) ---
            segments = [

    # ('2025-04-11 09:14:00', '2025-04-11 09:44:00',1),#冷凝盤管阻塞20%
    # ('2025-04-11 10:29:00', '2025-04-11 10:59:00',0),#蒸發盤管阻塞10%
    # ('2025-04-11 13:45:00', '2025-04-11 14:15:00',0),#蒸發風扇電流90%
    # ('2025-04-14 14:50:00', '2025-04-14 15:50:00',0),#冷媒洩漏20%
    # ('2026-01-01 00:00:00', '2026-01-01 00:30:00',0),#壓縮機故障10%
    # ('2026-01-01 02:00:00', '2026-01-01 02:30:00',0),#冷凝風扇電流上升10%
    # ('2026-01-01 04:00:00', '2026-01-01 04:30:00',0),#蒸發風扇電流上升10%

    # ('2025-04-14 10:45:00', '2025-04-14 11:15:00',0),#加熱器運轉
    # ('2026-01-01 06:20:00', '2026-01-01 06:40:00',0),#加熱器故障20%
                # {'name': '冷凝盤管20%', 'len': 1801},
                # {'name': '蒸發盤管10%', 'len': 1801},
                # {'name': '冷媒洩漏20%', 'len': 3601},
                # {'name': '壓縮機10%', 'len': 1801},
                # {'name': '冷凝風扇10%', 'len': 1801},
                # {'name': '蒸發風扇10%', 'len': 1801},

                {'name': '加熱器運轉', 'len': 1801},
                {'name': '加熱器故障20%', 'len': 1201},
                {'name': '加熱器故障30%', 'len': 1201},
            ]

            current_idx = 0
            y_min, y_max = plt.ylim()
            text_y_pos = y_max + (y_max - y_min) * 0.05 

            for seg in segments:
                start = current_idx
                end = current_idx + seg['len'] - 1
                if start >= n: break
                draw_end = min(end, n - 1)

                # 繪製區隔線
                if draw_end < n - 1:
                    plt.axvline(x=draw_end, color='black', linestyle='--', alpha=0.7)

                # 標示文字
                mid_point = (start + draw_end) / 2
                plt.text(mid_point, text_y_pos, seg['name'], 
                         ha='center', va='bottom', fontsize=12, fontweight='bold', color='black',
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

                current_idx += seg['len']

            plt.title(f'Predictions Test Source: y_score with Predicted Labels (Thr={best_thr:.3f})')
            plt.xlabel('Index (Time)')
            plt.ylabel('Score')
            plt.legend(loc='upper right')
            plt.tight_layout()
            
            output_png_name = 'predictions_test_source_marked_thr.png'
            output_png_path = os.path.join(results_dir, output_png_name)
            plt.savefig(output_png_path)
            plt.close()
            
            print(f"標記圖表已儲存至: {output_png_path}")
        else:
            print(f"找不到檔案: {predictions_csv}，無法繪製圖表。")

    except Exception as e:
        print(f"繪製圖表時發生錯誤: {e}")

    except Exception as e:
        print(f"繪製圖表時發生錯誤: {e}")

    except Exception as e:
        print(f"繪製自訂圖表時發生錯誤: {e}")

    print(f"======= HVAC 實驗全部完成 (src: {src}, trg: {trg}) =======")