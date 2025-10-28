import sys
# 使用絕對路徑強制 E:\DACAD 進入搜尋路徑
# 確保 E:\\DACAD 是您專案的正確路徑
sys.path.insert(0, 'D:\\DACAD')

import ast
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.augmentations import Injector # <-- 確保這一行在頂部
# ... (檔案的其餘部分保持不變)
from sklearn.model_selection import train_test_split

def get_dataset(args, domain_type, split_type):
    """
    Return the correct dataset object that will be fed into datalaoder
    args: args of main script
    domain_type: "source" or "target"
    split_type: "train" or "val" or "test"
    """
    
    if "SMD" in args.path_src:
        if domain_type == "source":
            return SMDDataset(args.path_src, subject_id=args.id_src, split_type=split_type, is_cuda=True)
        else:
            return SMDDataset_trg(args.path_trg, subject_id=args.id_trg, split_type=split_type, is_cuda=True)

    elif "MSL" in args.path_src:
        if domain_type == "source":
            return MSLDataset(args.path_src, subject_id=args.id_src, split_type=split_type, is_cuda=True)
        else:
            return MSLDataset_trg(args.path_trg, subject_id=args.id_trg, split_type=split_type, is_cuda=True)

    elif "Boiler" in args.path_src:
        if domain_type == "source":
            return BoilerDataset(args.path_src, subject_id=args.id_src, split_type=split_type, is_cuda=True)
        else:
            return BoilerDataset_trg(args.path_trg, subject_id=args.id_trg, split_type=split_type, is_cuda=True)
            
    elif "HVAC" in args.path_src: # <-- HVAC 區塊
        feature_columns = getattr(args, 'features', None) 
        if domain_type == "source":
            # 將 d_mean, d_std 傳遞給 Dataset
            return HVACDataset(args.path_src, subject_id=args.id_src, split_type=split_type, is_cuda=True,
                                feature_columns=feature_columns, d_mean=d_mean, d_std=d_std)
        else:
            # 將 d_mean, d_std 傳遞給 Dataset
            return HVACDataset_trg(args.path_trg, subject_id=args.id_trg, split_type=split_type, is_cuda=True,
                                feature_columns=feature_columns, d_mean=d_mean, d_std=d_std)

class MSLDataset(Dataset):
    def __init__(self, root_dir, subject_id, split_type="train", is_cuda=True, verbose=False):
        self.root_dir = root_dir
        self.subject_id = subject_id
        self.split_type = split_type
        self.is_cuda = is_cuda and torch.cuda.is_available()
        self.verbose = verbose

        self.load_sequence()

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, id_):
        sequence = self.sequence[id_]
        pid_ = np.random.randint(0, len(self.positive))
        positive = self.positive[pid_]
        random_choice = np.random.randint(0, 10)
        if random_choice == 0:
            nid_ = np.random.randint(0, len(self.negative))
            negative = self.negative[nid_]
        else:
            negative = get_injector(sequence, self.mean, self.std)

        # self.mean = None
        # self.std = None
        sequence_mask = np.ones(sequence.shape)
        label = self.label[id_]

        if self.is_cuda:
            sequence = torch.Tensor(sequence).float().cuda()
            sequence_mask = torch.Tensor(sequence_mask).long().cuda()
            positive = torch.Tensor(positive).float().cuda()
            negative = torch.Tensor(negative).float().cuda()
            label = torch.Tensor([label]).long().cuda()
        else:
            sequence = torch.Tensor(sequence).float()
            sequence_mask = torch.Tensor(sequence_mask).long()
            positive = torch.Tensor(positive).float()
            negative = torch.Tensor(negative).float()
            label = torch.Tensor([label]).long()

        sample = {"sequence": sequence, "sequence_mask": sequence_mask, "positive": positive, "negative": negative, "label": label}

        return sample

    def load_sequence(self):
        with open(os.path.join(self.root_dir, 'labeled_anomalies.csv'), 'r') as file:
            csv_reader = pd.read_csv(file, delimiter=',')

        # data_info = csv_reader[csv_reader['spacecraft'] == 'MSL']
        data_info = csv_reader[csv_reader['chan_id'] == self.subject_id]

        path_sequence = os.path.join(self.root_dir, 'test/', str(self.subject_id) + ".npy")
        temp = np.load(path_sequence)
        if np.any(sum(np.isnan(temp))!=0):
            print('Data contains NaN which replaced with zero')
            temp = np.nan_to_num(temp)

        self.mean = np.mean(temp, axis=0)
        self.std = np.std(temp, axis=0)
        self.std[self.std==0.0] = 1.0
        self.sequence = (temp - self.mean) / self.std

        labels = []
        for index, row in data_info.iterrows():
            anomalies = ast.literal_eval(row['anomaly_sequences'])
            length = row.iloc[-1]
            label = np.zeros([length], dtype=bool)
            for anomaly in anomalies:
                label[anomaly[0]:anomaly[1] + 1] = True
            labels.extend(label)
        self.label = np.asarray(labels)

        wsz, stride = 100, 1
        self.sequence , self.label = self.convert_to_windows(wsz, stride)
        self.positive = self.sequence[self.label == 1]
        self.negative = self.sequence[self.label == 0]

    def get_statistic(self):
        self.mean = np.mean(self.sequence, axis=0)
        self.std = np.std(self.sequence, axis=0)
        self.std[self.std==0.0] = 1.0
        return self.mean, self.std

    def convert_to_windows(self, w_size, stride):
        windows = []
        wlabels = []
        sz = int((self.sequence.shape[0]-w_size)/stride)
        for i in range(0, sz):
            st = i * stride
            w = self.sequence[st:st+w_size]
            if self.label[st:st+w_size].any() > 0:
                lbl = 1
            else: lbl=0
            windows.append(w)
            wlabels.append(lbl)
        return np.stack(windows), np.stack(wlabels)

class MSLDataset_trg(Dataset):
    def __init__(self, root_dir, subject_id, split_type="train", is_cuda=True, verbose=False):
        self.root_dir = root_dir
        self.subject_id = subject_id
        self.split_type = split_type
        self.is_cuda = is_cuda and torch.cuda.is_available()
        self.verbose = verbose

        self.load_sequence()

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, id_):
        sequence = self.sequence[id_]
        pid_ = abs(id_ - np.random.randint(1, 11))
        positive = self.sequence[pid_]
        self.positive = positive
        negative = get_injector(sequence, self.mean, self.std)
        self.negative = negative
        # self.mean = None
        # self.std = None
        sequence_mask = np.ones(sequence.shape)
        label = self.label[id_]

        if self.is_cuda:
            sequence = torch.Tensor(sequence).float().cuda()
            sequence_mask = torch.Tensor(sequence_mask).long().cuda()
            positive = torch.Tensor(positive).float().cuda()
            negative = torch.Tensor(negative).float().cuda()
            label = torch.Tensor([label]).long().cuda()
        else:
            sequence = torch.Tensor(sequence).float()
            sequence_mask = torch.Tensor(sequence_mask).long()
            positive = torch.Tensor(positive).float()
            negative = torch.Tensor(negative).float()
            label = torch.Tensor([label]).long()

        sample = {"sequence": sequence, "sequence_mask": sequence_mask, "positive": positive, "negative": negative, "label": label}

        return sample

    def load_sequence(self):
        with open(os.path.join(self.root_dir, 'labeled_anomalies.csv'), 'r') as file:
            csv_reader = pd.read_csv(file, delimiter=',')

        # data_info = csv_reader[csv_reader['spacecraft'] == 'MSL']
        data_info = csv_reader[csv_reader['chan_id'] == self.subject_id]

        path_sequence = os.path.join(self.root_dir, 'test/', str(self.subject_id) + ".npy")
        temp = np.load(path_sequence)
        if np.any(sum(np.isnan(temp))!=0):
            print('Data contains NaN which replaced with zero')
            temp = np.nan_to_num(temp)

        self.mean = np.mean(temp, axis=0)
        self.std = np.std(temp, axis=0)
        self.std[self.std==0.0] = 1.0
        self.sequence = (temp - self.mean) / self.std

        labels = []
        for index, row in data_info.iterrows():
            anomalies = ast.literal_eval(row['anomaly_sequences'])
            length = row.iloc[-1]
            label = np.zeros([length], dtype=bool)
            for anomaly in anomalies:
                label[anomaly[0]:anomaly[1] + 1] = True
            labels.extend(label)
        self.label = np.asarray(labels)

        wsz, stride = 100, 1
        self.sequence , self.label = self.convert_to_windows(wsz, stride)
        self.positive = self.sequence[self.label == 1]
        self.negative = self.sequence[self.label == 0]

    def get_statistic(self):
        self.mean = np.mean(self.sequence, axis=0)
        self.std = np.std(self.sequence, axis=0)
        self.std[self.std==0.0] = 1.0
        return self.mean, self.std

    def convert_to_windows(self, w_size, stride):
        windows = []
        wlabels = []
        sz = int((self.sequence.shape[0] - w_size) / stride)
        for i in range(0, sz):
            st = i * stride
            w = self.sequence[st:st + w_size]
            if self.label[st:st + w_size].any() > 0:
                lbl = 1
            else:
                lbl = 0
            windows.append(w)
            wlabels.append(lbl)
        return np.stack(windows), np.stack(wlabels)

class SMDDataset(Dataset):
    def __init__(self, root_dir, subject_id, split_type="train", is_cuda=False, verbose=False):
        self.root_dir = root_dir
        self.subject_id = subject_id
        self.split_type = split_type
        self.is_cuda = is_cuda and torch.cuda.is_available()
        self.verbose = verbose

        self.load_sequence()

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, id_):
        sequence = self.sequence[id_]
        pid_ = np.random.randint(0, len(self.positive))
        positive = self.positive[pid_]
        random_choice = np.random.randint(0, 10)
        if random_choice == 0:
            nid_ = np.random.randint(0, len(self.negative))
            negative = self.negative[nid_]
        else:
            negative = get_injector(sequence, self.mean, self.std)

        sequence_mask = np.ones(sequence.shape)
        label = self.label[id_]

        if self.is_cuda:
            sequence = torch.Tensor(sequence).float().cuda()
            sequence_mask = torch.Tensor(sequence_mask).long().cuda()
            positive = torch.Tensor(positive).float().cuda()
            negative = torch.Tensor(negative).float().cuda()
            label = torch.Tensor([label]).long().cuda()
        else:
            sequence = torch.Tensor(sequence).to(torch.float32)
            sequence_mask = torch.Tensor(sequence_mask).long()
            positive = torch.Tensor(positive).to(torch.float32)
            negative = torch.Tensor(negative).to(torch.float32)
            label = torch.Tensor([label]).long()

        sample = {"sequence": sequence, "sequence_mask": sequence_mask, "positive": positive, "negative": negative, "label": label}

        return sample

    def load_sequence(self):
        path_sequence = os.path.join(self.root_dir, "machine-" + str(self.subject_id) + ".txt")
        self.sequence = np.loadtxt(path_sequence, delimiter=",")

        # if self.split_type == "test":
        self.mean = np.mean(self.sequence, axis=0)
        self.std = np.std(self.sequence, axis=0)
        self.std[self.std==0.0] = 1.0
        self.sequence = (self.sequence - self.mean) / self.std

        path_label = os.path.join(self.root_dir+ "_label", "machine-" + str(self.subject_id) + ".txt")
        self.label = np.loadtxt(path_label)

        wsz, stride = 100, 1
        self.sequence , self.label = self.convert_to_windows(wsz, stride)
        self.positive = self.sequence[self.label == 1]
        self.negative = self.sequence[self.label == 0]

    def get_statistic(self):
        self.mean = np.mean(self.sequence, axis=0)
        self.std = np.std(self.sequence, axis=0)
        self.std[self.std==0.0] = 1.0
        return self.mean, self.std

    def convert_to_windows(self, w_size, stride):
        windows = []
        wlabels = []
        sz = int((self.sequence.shape[0]-w_size)/stride)
        for i in range(0, sz):
            st = i * stride
            w = self.sequence[st:st+w_size]
            if self.label[st:st+w_size].any() > 0:
                lbl = 1
            else: lbl=0
            windows.append(w)
            wlabels.append(lbl)
        return np.stack(windows), np.stack(wlabels)

class SMDDataset_trg(Dataset):
    def __init__(self, root_dir, subject_id, split_type="train", is_cuda=True, verbose=False):
        self.root_dir = root_dir
        self.subject_id = subject_id
        self.split_type = split_type
        self.is_cuda = is_cuda and torch.cuda.is_available()
        self.verbose = verbose

        self.load_sequence()

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, id_):
        sequence = self.sequence[id_]
        pid_ = abs(id_ - np.random.randint(1, 11))
        positive = self.sequence[pid_]
        self.positive = positive
        negative = get_injector(sequence, self.mean, self.std)
        self.negative = negative

        sequence_mask = np.ones(sequence.shape)
        label = self.label[id_]

        if self.is_cuda:
            sequence = torch.Tensor(sequence).float().cuda()
            sequence_mask = torch.Tensor(sequence_mask).long().cuda()
            positive = torch.Tensor(positive).float().cuda()
            negative = torch.Tensor(negative).float().cuda()
            label = torch.Tensor([label]).long().cuda()
        else:
            sequence = torch.Tensor(sequence).to(torch.float32)
            sequence_mask = torch.Tensor(sequence_mask).long()
            positive = torch.Tensor(positive).to(torch.float32)
            negative = torch.Tensor(negative).to(torch.float32)
            label = torch.Tensor([label]).long()

        sample = {"sequence": sequence, "sequence_mask": sequence_mask, "positive": positive, "negative": negative, "label": label}

        return sample

    def load_sequence(self):
        path_sequence = os.path.join(self.root_dir, "machine-" + str(self.subject_id) + ".txt")
        self.sequence = np.loadtxt(path_sequence, delimiter=",")

        # if self.split_type == "test":
        self.mean = np.mean(self.sequence, axis=0)
        self.std = np.std(self.sequence, axis=0)
        self.std[self.std==0.0] = 1.0
        self.sequence = (self.sequence - self.mean) / self.std

        path_label = os.path.join(self.root_dir+ "_label", "machine-" + str(self.subject_id) + ".txt")
        self.label = np.loadtxt(path_label)

        wsz, stride = 100, 1
        self.sequence , self.label = self.convert_to_windows(wsz, stride)
        self.positive = self.sequence[self.label == 1]
        self.negative = self.sequence[self.label == 0]

    def get_statistic(self):
        self.mean = np.mean(self.sequence, axis=0)
        self.std = np.std(self.sequence, axis=0)
        self.std[self.std==0.0] = 1.0
        return self.mean, self.std

    def convert_to_windows(self, w_size, stride):
        windows = []
        wlabels = []
        sz = int((self.sequence.shape[0] - w_size) / stride)
        for i in range(0, sz):
            st = i * stride
            w = self.sequence[st:st + w_size]
            if self.label[st:st + w_size].any() > 0:
                lbl = 1
            else:
                lbl = 0
            windows.append(w)
            wlabels.append(lbl)
        return np.stack(windows), np.stack(wlabels)

class BoilerDataset(Dataset):
    def __init__(self, root_dir, subject_id, split_type="train", is_cuda=True, verbose=False):
        self.root_dir = root_dir
        self.subject_id = subject_id
        self.split_type = split_type
        self.is_cuda = is_cuda and torch.cuda.is_available()
        self.verbose = verbose

        self.load_sequence()

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, id_):
        sequence = self.sequence[id_]
        pid_ = np.random.randint(0, len(self.positive))
        positive = self.positive[pid_]
        random_choice = np.random.randint(0, 10)
        if random_choice == 0:
            nid_ = np.random.randint(0, len(self.negative))
            negative = self.negative[nid_]
        else:
            negative = get_injector(sequence, self.mean, self.std)

        sequence_mask = np.ones(sequence.shape)
        label = self.label[id_]

        if self.is_cuda:
            sequence = torch.Tensor(sequence).float().cuda()
            sequence_mask = torch.Tensor(sequence_mask).long().cuda()
            positive = torch.Tensor(positive).float().cuda()
            negative = torch.Tensor(negative).float().cuda()
            label = torch.Tensor([label]).long().cuda()
        else:
            sequence = torch.Tensor(sequence).float()
            sequence_mask = torch.Tensor(sequence_mask).long()
            positive = torch.Tensor(positive).float()
            negative = torch.Tensor(negative).float()
            label = torch.Tensor([label]).long()

        sample = {"sequence": sequence, "sequence_mask": sequence_mask, "positive": positive, "negative": negative, "label": label}

        return sample

    def load_sequence(self):
        path_sequence = os.path.join(self.root_dir, (self.subject_id) + ".csv")
        self.sequence = pd.read_csv(path_sequence).values
        self.label = self.sequence[:, -1]
        self.sequence = self.sequence[:, 2:-1].astype(float)

        # if self.split_type == "test":
        self.mean = np.mean(self.sequence, axis=0)
        self.std = np.std(self.sequence, axis=0)
        self.std[self.std==0.0] = 1.0
        self.sequence = (self.sequence - self.mean) / self.std

        wsz, stride = 100, 1
        self.sequence , self.label = self.convert_to_windows(wsz, stride)
        self.positive = self.sequence[self.label == 1]
        self.negative = self.sequence[self.label == 0]

    def get_statistic(self):
        self.mean = np.mean(self.sequence, axis=0)
        self.std = np.std(self.sequence, axis=0)
        self.std[self.std==0.0] = 1.0
        return self.mean, self.std

    def convert_to_windows(self, w_size, stride):
        windows = []
        wlabels = []
        sz = int((self.sequence.shape[0]-w_size)/stride)
        for i in range(0, sz):
            st = i * stride
            w = self.sequence[st:st+w_size]
            if self.label[st:st+w_size].any() > 0:
                lbl = 1
            else: lbl=0
            windows.append(w)
            wlabels.append(lbl)
        return np.stack(windows), np.stack(wlabels)

class BoilerDataset_trg(Dataset):
    def __init__(self, root_dir, subject_id, split_type="train", is_cuda=True, verbose=False):
        self.root_dir = root_dir
        self.subject_id = subject_id
        self.split_type = split_type
        self.is_cuda = is_cuda and torch.cuda.is_available()
        self.verbose = verbose

        self.load_sequence()

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, id_):
        sequence = self.sequence[id_]
        pid_ = abs(id_ - np.random.randint(1, 11))
        positive = self.sequence[pid_]
        self.positive = positive
        negative = get_injector(sequence, self.mean, self.std)
        self.negative = negative

        sequence_mask = np.ones(sequence.shape)
        label = self.label[id_]

        if self.is_cuda:
            sequence = torch.Tensor(sequence).float().cuda()
            sequence_mask = torch.Tensor(sequence_mask).long().cuda()
            positive = torch.Tensor(positive).float().cuda()
            negative = torch.Tensor(negative).float().cuda()
            label = torch.Tensor([label]).long().cuda()
        else:
            sequence = torch.Tensor(sequence).float()
            sequence_mask = torch.Tensor(sequence_mask).long()
            positive = torch.Tensor(positive).float()
            negative = torch.Tensor(negative).float()
            label = torch.Tensor([label]).long()

        sample = {"sequence": sequence, "sequence_mask": sequence_mask, "positive": positive, "negative": negative, "label": label}

        return sample

    def load_sequence(self):
        path_sequence = os.path.join(self.root_dir, (self.subject_id) + ".csv")
        self.sequence = pd.read_csv(path_sequence).values
        self.label = self.sequence[:, -1]
        self.sequence = self.sequence[:, 2:-1].astype(float)

        # if self.split_type == "test":
        self.mean = np.mean(self.sequence, axis=0)
        self.std = np.std(self.sequence, axis=0)
        self.std[self.std==0.0] = 1.0
        self.sequence = (self.sequence - self.mean) / self.std

        wsz, stride = 100, 1
        self.sequence , self.label = self.convert_to_windows(wsz, stride)
        self.positive = self.sequence[self.label == 1]
        self.negative = self.sequence[self.label == 0]

    def get_statistic(self):
        self.mean = np.mean(self.sequence, axis=0)
        self.std = np.std(self.sequence, axis=0)
        self.std[self.std==0.0] = 1.0
        return self.mean, self.std

    def convert_to_windows(self, w_size, stride):
        windows = []
        wlabels = []
        sz = int((self.sequence.shape[0]-w_size)/stride)
        for i in range(0, sz):
            st = i * stride
            w = self.sequence[st:st+w_size]
            if self.label[st:st+w_size].any() > 0:
                lbl = 1
            else: lbl=0
            windows.append(w)
            wlabels.append(lbl)
        return np.stack(windows), np.stack(wlabels)

# =============================================================================
# START: 修改 HVAC 類別
# =============================================================================
class HVACDataset(Dataset):
    def __init__(self, root_dir, subject_id, split_type="train", is_cuda=True, verbose=False, 
                 feature_columns=None, w_size=100, stride=1, 
                 d_mean=None, d_std=None): # <-- 1. 新增 d_mean, d_std 參數
        
        self.root_dir = root_dir
        self.subject_id = subject_id
        self.split_type = split_type
        self.is_cuda = is_cuda and torch.cuda.is_available()
        self.verbose = verbose
        self.feature_columns = feature_columns
        self.w = w_size
        self.s = stride

        # 傳入的 d_mean, d_std
        self.d_mean = d_mean
        self.d_std = d_std
        
        self.sequence = None
        self.label = None

        self.load_sequence() # 載入並處理資料
        
        self.sequence , self.label = self.convert_to_windows(self.w, self.s)
        
        # 你的 val 檔案現在保證有 0 和 1, 不會再崩潰
        self.positive = self.sequence[self.label == 1]
        self.negative = self.sequence[self.label == 0]

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, id_):
        # 使用我們上次修正過的 __getitem__ 邏輯，以防萬一
        sequence = self.sequence[id_]
        pid_ = np.random.randint(0, len(self.positive))
        positive = self.positive[pid_]

        random_choice = np.random.randint(0, 10)
        if random_choice == 0 and len(self.negative) > 0:
            nid_ = np.random.randint(0, len(self.negative))
            negative = self.negative[nid_]
        else:
            negative = get_injector(sequence, self.d_mean, self.d_std) # 使用 self.d_mean

        sequence_mask = np.ones(sequence.shape)
        label = self.label[id_]

        if self.is_cuda:
            # ... (torch 轉換邏輯不變) ...
            sequence = torch.Tensor(sequence).float().cuda()
            sequence_mask = torch.Tensor(sequence_mask).long().cuda()
            positive = torch.Tensor(positive).float().cuda()
            negative = torch.Tensor(negative).float().cuda()
            label = torch.Tensor([label]).long().cuda()
        else:
            sequence = torch.Tensor(sequence).float()
            sequence_mask = torch.Tensor(sequence_mask).long()
            positive = torch.Tensor(positive).float()
            negative = torch.Tensor(negative).float()
            label = torch.Tensor([label]).long()

        sample = {"sequence": sequence, "sequence_mask": sequence_mask, "positive": positive, "negative": negative, "label": label}
        return sample

    def load_sequence(self):
        # --- START: 修正邏輯 ---
        
        # 2. 根據 split_type 決定檔名
        # 假設你的檔案名稱是 "source_data_train.csv", "source_data_val.csv"
        # 你的 subject_id 可能是 "source_data"
        
        if self.split_type == "train":
            filename = f"{self.subject_id}_train.csv"
        elif self.split_type == "val":
            filename = f"{self.subject_id}_val.csv"
        elif self.split_type == "test":
            filename = f"{self.subject_id}.csv" # 或者 "test.csv"，取決於 eval.py 的呼叫
            # 為了安全起見，如果 eval.py 是用 "source_data" 和 "test" 來呼叫，我們就讀 "test.csv"
            # 讓我們假設 eval.py 呼叫的 subject_id 是 "test"
            if self.subject_id == "test":
                 filename = "test.csv"
            # **** 注意：這裡的邏輯高度依賴你的檔名和 eval.py ****
            # **** 為了簡化，我們假設 "test" 模式就是讀 "test.csv" ****
            # **** 並且 main_HVAC.py 中 args.id_trg 在 test 模式下是 "test" ****
            #
            # 我們採用更穩健的假設：
            # split_type="test" 時，我們就讀 "test.csv" (忽略 subject_id)
            # 你必須確保 `main/eval.py` 呼叫 `get_dataset` 時 `split_type="test"`
            
            # --- 讓我們重新定義檔名邏輯 ---
            if self.subject_id == "test": # 假設 eval.py 會傳 "test"
                filename = "test.csv"
            else:
                # 假設你的檔名是 source_data_train.csv, target_data_train.csv ...
                filename = f"{self.subject_id}_{self.split_type}.csv"
        
        path_sequence = os.path.join(self.root_dir, filename)
        if self.verbose: print(f"[HVACDataset] Loading file: {path_sequence}")
        
        try:
            df = pd.read_csv(path_sequence)
        except FileNotFoundError:
            print(f"錯誤：找不到檔案 {path_sequence}")
            print("請確保你的手動分割檔案名稱符合 {subject_id}_{split_type}.csv 格式")
            print(f"(例如: source_data_train.csv, source_data_val.csv)")
            raise

        # 3. 決定使用哪些特徵欄位 (邏輯不變)
        cols_to_use = []
        if self.feature_columns:
            cols_to_use = self.feature_columns
        else:
            cols_to_use = df.columns[2:-1]

        features = df[cols_to_use].astype(float)
        self.label = df.iloc[:, -1].values

        # 4. 【關鍵】處理標準化
        if self.d_mean is None:
            # 如果沒有傳入 mean/std (只會在 source_train 時發生)
            if self.verbose: print(f"[HVACDataset] Calculating new mean/std from {filename}")
            self.d_mean = np.mean(features.values, axis=0)
            self.d_std = np.std(features.values, axis=0)
            self.d_std[self.d_std==0.0] = 1.0
        else:
            # 如果有傳入 mean/std (val/test/target 時發生)
            if self.verbose: print(f"[HVACDataset] Using provided mean/std.")
            pass # self.d_mean 和 self.d_std 已經被 __init__ 設定

        # 5. 標準化
        self.sequence = (features - self.d_mean) / self.d_std
        self.sequence = self.sequence.values
            
        # --- END: 修正邏輯 ---

    def get_statistic(self):
        # 這個函式現在回傳的是【訓練集】的統計數據
        return self.d_mean, self.d_std

    def convert_to_windows(self, w_size, stride):
        # ... (此函式不變) ...
        windows = []
        wlabels = []
        sz = int((self.sequence.shape[0]-w_size)/stride)
        for i in range(0, sz):
            st = i * stride
            w = self.sequence[st:st+w_size]
            if self.label[st:st+w_size].any() > 0:
                lbl = 1
            else: lbl=0
            windows.append(w)
            wlabels.append(lbl)
        return np.stack(windows), np.stack(wlabels)

class HVACDataset_trg(Dataset):
    def __init__(self, root_dir, subject_id, split_type="train", is_cuda=True, verbose=False,
                 feature_columns=None, w_size=100, stride=1,
                 d_mean=None, d_std=None): # <-- 1. 新增 d_mean, d_std 參數
        
        self.root_dir = root_dir
        self.subject_id = subject_id
        self.split_type = split_type
        self.is_cuda = is_cuda and torch.cuda.is_available()
        self.verbose = verbose
        self.feature_columns = feature_columns
        self.w = w_size
        self.s = stride

        # 傳入的 d_mean, d_std
        self.d_mean = d_mean
        self.d_std = d_std
        
        self.sequence = None
        self.label = None

        self.load_sequence() # 載入並處理資料
        
        self.sequence , self.label = self.convert_to_windows(self.w, self.s)
        self.positive = self.sequence[self.label == 1]
        self.negative = self.sequence[self.label == 0]

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, id_):
        # ... ( __getitem__ 邏輯不變 ) ...
        sequence = self.sequence[id_]
        pid_ = abs(id_ - np.random.randint(1, 11))
        positive = self.sequence[pid_]
        self.positive = positive
        negative = get_injector(sequence, self.d_mean, self.d_std) # 使用 self.d_mean
        self.negative = negative

        sequence_mask = np.ones(sequence.shape)
        label = self.label[id_]

        if self.is_cuda:
            sequence = torch.Tensor(sequence).float().cuda()
            sequence_mask = torch.Tensor(sequence_mask).long().cuda()
            positive = torch.Tensor(positive).float().cuda()
            negative = torch.Tensor(negative).float().cuda()
            label = torch.Tensor([label]).long().cuda()
        else:
            sequence = torch.Tensor(sequence).float()
            sequence_mask = torch.Tensor(sequence_mask).long()
            positive = torch.Tensor(positive).float()
            negative = torch.Tensor(negative).float()
            label = torch.Tensor([label]).long()

        sample = {"sequence": sequence, "sequence_mask": sequence_mask, "positive": positive, "negative": negative, "label": label}
        return sample

    def load_sequence(self):
        # --- START: 修正邏輯 ---
        
        # 2. 根據 split_type 決定檔名
        # 假設你的檔名是 target_data_train.csv, target_data_val.csv
        filename = f"{self.subject_id}_{self.split_type}.csv"
        
        path_sequence = os.path.join(self.root_dir, filename)
        if self.verbose: print(f"[HVACDataset_trg] Loading file: {path_sequence}")
        
        try:
            df = pd.read_csv(path_sequence)
        except FileNotFoundError:
            print(f"錯誤：找不到檔案 {path_sequence}")
            print("請確保你的手動分割檔案名稱符合 {subject_id}_{split_type}.csv 格式")
            print(f"(例如: target_data_train.csv, target_data_val.csv)")
            raise

        # 3. 決定使用哪些特徵欄位 (邏輯不變)
        cols_to_use = []
        if self.feature_columns:
            cols_to_use = self.feature_columns
        else:
            cols_to_use = df.columns[2:-1]

        features = df[cols_to_use].astype(float)
        self.label = df.iloc[:, -1].values

        # 4. 【關鍵】處理標準化
        if self.d_mean is None:
            # target_train 應該【總是】使用 source_train 的 mean/std
            # 所以這裡【不應該】計算新的
            if self.verbose: print(f"[HVACDataset_trg] Warning: d_mean is None. Using self-calculated mean/std.")
            self.d_mean = np.mean(features.values, axis=0)
            self.d_std = np.std(features.values, axis=0)
            self.d_std[self.d_std==0.0] = 1.0
        else:
            if self.verbose: print(f"[HVACDataset_trg] Using provided mean/std.")
            pass # self.d_mean 和 self.d_std 已經被 __init__ 設定

        # 5. 標準化
        self.sequence = (features - self.d_mean) / self.d_std
        self.sequence = self.sequence.values
            
        # --- END: 修正邏輯 ---

    def get_statistic(self):
        return self.d_mean, self.d_std

    def convert_to_windows(self, w_size, stride):
        # ... (此函式不變) ...
        windows = []
        wlabels = []
        sz = int((self.sequence.shape[0]-w_size)/stride)
        for i in range(0, sz):
            st = i * stride
            w = self.sequence[st:st+w_size]
            if self.label[st:st+w_size].any() > 0:
                lbl = 1
            else: lbl=0
            windows.append(w)
            wlabels.append(lbl)
        return np.stack(windows), np.stack(wlabels)
# =============================================================================
# END: 修改 HVAC 類別
# =============================================================================

def get_injector(sample_batched, d_mean, d_std):
    sample_batched = (sample_batched * d_std) + d_mean
    injected_window = Injector(sample_batched)
    injected_window.injected_win = (injected_window.injected_win - d_mean) / d_std

    return injected_window.injected_win


def get_output_dim(args):
    output_dim = -1

    if "SMD" in args.path_src:
        output_dim = 1
    elif "MSL" in args.path_src:
        output_dim = 1
    elif "Boiler" in args.path_src:
        output_dim = 1
    elif "HVAC" in args.path_src: # <-- 新增 HVAC 
        output_dim = 1
    else:
        output_dim = 6

    return output_dim

def collate_test(batch):
    #The input is list of dictionaries
    out = {}
    for key in batch[0].keys():
        val = []
        for sample in batch:
            val.append(sample[key])
        val = torch.cat(val, dim=0)
        out[key] = val
    return out