from import_library import *

class CycleDataset(Dataset):
    def __init__(self, data, max_duration, fs, S, C):
        self.data = data
        self.max_duration = max_duration
        self.fs = fs
        self.S = S
        self.C = C

    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        data_raw = self.data.iloc[idx]['data']
        label_matrix = self.data.iloc[idx]['inhale_range']
        return data_raw, label_matrix