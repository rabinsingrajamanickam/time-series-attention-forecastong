import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

class TimeSeriesDataset(Dataset):
    def __init__(self, file, seq_len, pred_len):
        df = pd.read_csv(file)
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(df.values)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+self.seq_len:idx+self.seq_len+self.pred_len,0]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
