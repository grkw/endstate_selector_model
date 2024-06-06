import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from csv_dataset import CSVDataset
from endstate_selector_model import FCNStateSelector, CNNStateSelector, LSTMStateSelector

num_vf_angles = 18
num_vf_mags = 2
num_vf_choices = (num_vf_angles)*(num_vf_mags+1)
num_af_angles = 19
num_af_mags = 2
num_af_choices = (num_af_angles)*(num_af_mags+1)

# Example usage
csv_file = 'path/to/your/data.csv'
batch_size = 4

dataset = CSVDataset(csv_file)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Iterate through the DataLoader
for i, (inputs, labels) in enumerate(dataloader):
    print(f'Batch {i+1}')
    print('Inputs:', inputs)
    print('Labels:', labels)

# Initialize and train your model (example)
model = FCNStateSelector()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy training loop
for epoch in range(100):
    optimizer.zero_grad()
    inputs = torch.randn(1, 10)
    outputs = model(inputs)
    loss = criterion(outputs, torch.randn(1, 1))
    loss.backward()
    optimizer.step()

# Save the trained model
torch.save(model.state_dict(), "simple_model.pth")

# Convert to TorchScript using tracing
example_input = torch.randn(1, 10)
traced_script_module = torch.jit.trace(model, example_input)
traced_script_module.save("simple_model.pt")