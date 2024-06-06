import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class CSVDataset(Dataset):
    def __init__(self, csv_file, input_col, label_col, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        self.le = LabelEncoder()
        self.data.iloc[:, label_col] = self.le.fit_transform(self.data.iloc[:, label_col])

        self.label_col = label_col
        self.input_col = input_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Assuming the first 9 columns are the input features
        inputs = self.data.iloc[idx, :self.input_col].values.astype('float32')
        
        # Assuming the next 4 columns are the output labels
        outputs = self.data.iloc[idx, self.label_col].values.astype('int32')

        if self.transform:
            sample = self.transform(sample)

        return inputs, outputs
