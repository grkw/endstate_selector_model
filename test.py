import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import wandb

from csv_dataset import CSVDataset
from endstate_selector_model import FCNStateSelector, CNNStateSelector, LSTMStateSelector
from config import Config

cfg = Config()

# Model test settings
test_csv_file = 'data/test_64paths_3wps_srand1.csv'
batch_size = 4

# Load the test data
test_dataset = CSVDataset(test_csv_file, cfg.csv_input_col, cfg.csv_label_col)  # replace with your test csv file
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Load the best model
best_model = FCNStateSelector(cfg.input_size, cfg.output_size)
best_model.load_state_dict(torch.load("models/best_model.pth"))

# Evaluate the model
criterion = nn.CrossEntropyLoss()
best_model.eval()
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for batch, (test_inputs, test_labels) in enumerate(test_dataloader):
        outputs = best_model(test_inputs)

        # Calculate loss
        loss = criterion(outputs, test_labels)
        test_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += test_labels.size(0)
        correct += (predicted == test_labels).sum().item()
        print('Predicted', predicted)
        print('Truth', test_labels)

print('Total # of test examples: ', total)
print('# of correctly predicted examples: ', correct)

print(f'Test Loss: {test_loss / len(test_dataloader)}')
print(f'Test Accuracy: {correct / total * 100}%')
print(f'Guessing Accuracy: {1 / cfg.num_vf_choices * 100}%')