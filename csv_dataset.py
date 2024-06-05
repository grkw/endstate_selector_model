import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class CSVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Assuming the first 9 columns are the input features
        inputs = self.data.iloc[idx, :9].values.astype('float32')
        
        # Assuming the next 4 columns are the output labels
        outputs = self.data.iloc[idx, 9:13].values.astype('float32')

        if self.transform:
            sample = self.transform(sample)

        return inputs, outputs
