import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import wandb

from csv_dataset import CSVDataset, CSVDatasetMSE
from endstate_selector_model import FCNStateSelector, CNNStateSelector, LSTMStateSelector
from config import Config
import numpy as np

def find_closest_vector(target_vector):
    # target_vector = target_vector.reshape(64, 1, 2)
    array_of_vectors = torch.Tensor([[-1.00,0.00],[10.00,2.50],[10.00,5.00],[100.00,2.50],[100.00,5.00],[110.00,2.50],[110.00,5.00],[120.00,2.50],[120.00,5.00],[130.00,2.50],[130.00,5.00],[140.00,2.50],[140.00,5.00],[150.00,2.50],[150.00,5.00],[160.00,2.50],[160.00,5.00],[170.00,2.50],[170.00,5.00],[180.00,2.50],[180.00,5.00],[20.00,2.50],[20.00,5.00],[30.00,2.50],[30.00,5.00],[40.00,2.50],[40.00,5.00],[50.00,2.50],[50.00,5.00],[60.00,2.50],[60.00,5.00],[70.00,2.50],[70.00,5.00],[80.00,2.50],[80.00,5.00],[90.00,2.50],[90.00,5.00]])
    # array_of_vectors = array_of_vectors.reshape(1,37,2)

    closest_indeces = torch.zeros(64)
    for idx in range(64):
        distances = torch.norm(array_of_vectors - target_vector[idx], dim=1, keepdim=True)
        # print('distances.shape: ', distances.shape)
        # print('distances: ', distances)
        closest_index = torch.argmin(distances, dim=0)
        # print("closest_index: ", closest_index)
        closest_indeces[idx] = closest_index
        # print('closest_index.shape: ', closest_indeces.shape)
    return closest_indeces

cfg = Config()

# Load the test data
test_dataset = CSVDataset(cfg.test_csv_file, cfg.csv_input_col, cfg.vf_desc_col)  # replace with your test csv file
test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4)

# Load the best model
best_model = FCNStateSelector(cfg.input_size, cfg.output_size)
best_model.load_state_dict(torch.load(cfg.best_model_path))

# Evaluate the model
criterion = nn.MSELoss()#nn.CrossEntropyLoss()
best_model.eval()
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for batch, (test_inputs, test_labels) in enumerate(test_dataloader):
        if test_labels.shape[0] != cfg.batch_size:
            continue

        outputs = best_model(test_inputs)
        # print('outputs.shape: ', outputs.shape)
        # Calculate loss
        # loss = criterion(outputs, test_labels)
        # test_loss += loss.item()

        # Calculate accuracy
        predicted = find_closest_vector(outputs.data)#torch.max(outputs.data, 1)
        total += test_labels.size(0)
        # print("predicted.shape: ", predicted.shape)
        # print("test_labels.shape: ", test_labels.shape)
        correct += (predicted == test_labels).sum().item()
        print('Predicted', predicted)
        print('Truth', test_labels)

print('Total # of test examples: ', total)
print('# of correctly predicted examples: ', correct)

print(f'Test Loss: {round(test_loss / len(test_dataloader),3)}')
print(f'Test Accuracy: {round(correct / total * 100,3)}%')
print(f'Guessing Accuracy: {round(1 / cfg.num_vf_choices * 100,3)}%')