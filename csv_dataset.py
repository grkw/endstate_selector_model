import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

class CSVDataset(Dataset):
    def __init__(self, csv_file, input_col, label_col, transform=None):
        self.data = pd.read_csv(csv_file) # automatically treats the first row as the header and does not include it in the data
        print("Data shape: ", self.data.shape)
        self.transform = transform

        self.le = LabelEncoder()
        self.data.iloc[:, label_col] = self.le.fit_transform(self.data.iloc[:, label_col])
        print("LabelEncoder classes: \n", self.le.classes_)
        print("LabelEncoder classes shape: ", self.le.classes_.shape)

        labels = self.data.iloc[:, label_col]
        label_distribution = np.bincount(labels)
        print("Label distribution: ", label_distribution)

        self.label_col = label_col
        self.input_col = input_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = self.data.iloc[idx, :self.input_col]
        inputs = torch.tensor(inputs, dtype=torch.float32)
        
        outputs = self.data.iloc[idx, self.label_col]
        outputs = torch.tensor(outputs, dtype=torch.long)

        if self.transform:
            sample = self.transform(sample)

        return inputs, outputs
