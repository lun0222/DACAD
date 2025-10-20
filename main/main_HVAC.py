import os
import subprocess
import sys

if __name__ == '__main__':
    # 獲取 main_HVAC.py 所在的目錄 (例如 e:\DACAD\main)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 獲取專案根目錄 (例如 e:\DACAD)
    project_root = os.path.dirname(current_dir)
    
    # 1. 設定您的 HVAC 數據集路徑
    dataset_path = os.path.join(project_root, 'datasets', 'HVAC')
    
    # 檢查路徑是否存在
    if not os.path.exists(dataset_path):
        print(f"錯誤：找不到數據集路徑 {dataset_path}")
        print(f"請確保您的 HVAC 數據 (例如 source_data.csv) 位於 {os.path.join(project_root, 'datasets', 'HVAC')}")
        sys.exit()
        
    all_files = os.listdir(dataset_path)

    # 2. 找到所有 .csv 檔案
    files = [name[:-4] for name in all_files if name.endswith('.csv')]
    files = sorted(files)
    
    # 3. 獲取 train.py 和 eval.py 的絕對路徑
    train_script = os.path.join(current_dir, 'train.py')
    eval_script = os.path.join(current_dir, 'eval.py')
    
    # 獲取當前正在使用的 Python 解譯器路徑
    python_executable = sys.executable
    
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
                    '--num_epochs', '20',
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
                    '--momentum', '0.99'
                ]
                
                # 在預設目錄 (E:\DACAD) 中執行
                subprocess.run(command_train)

                # 6. 定義評估命令
                command_eval = [
                    python_executable, eval_script,
                    '--experiments_main_folder', 'results',
                    '--experiment_folder', 'HVAC',
                    '--id_src', src,
                    '--id_trg', trg
                ]
                
                # 在預設目錄 (E:\DACAD) 中執行
                subprocess.run(command_eval)

    print("======= 所有 HVAC 實驗已完成 =======")