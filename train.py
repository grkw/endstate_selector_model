import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from csv_dataset import CSVDataset
from endstate_selector_model import FCNStateSelector, CNNStateSelector, LSTMStateSelector
import wandb

# Planner settings
num_waypoints = 3

num_vf_angles = 18
num_vf_mags = 2
num_vf_choices = (num_vf_angles)*(num_vf_mags+1)
num_af_angles = 19
num_af_mags = 2
num_af_choices = (num_af_angles)*(num_af_mags+1)

# Model settings
train_csv_file = 'data/train_64paths_3wps_srand4.csv'
val_csv_file = 'data/val_64paths_3wps_srand3.csv'
batch_size = 64
input_size = 3*num_waypoints + 9
output_size = num_vf_choices

train_dataset = CSVDataset(train_csv_file)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

val_dataset = CSVDataset(val_csv_file)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Might wanna run this on Colab so I don't have to keep loading the dataset and I can use my compute units

wandb.init(project="endstate-selector")

# Initialize and train your model (example)
model = FCNStateSelector()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

best_val_loss = float('inf')

for epoch in range(100):
    
    model.train()
    running_loss = 0.0
    for batch, (train_inputs, train_labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(train_inputs)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    wandb.log({"train_loss": running_loss})
    print(f"Epoch {epoch + 1}, loss: {running_loss / len(train_dataloader)}")
          
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch, (val_inputs, val_labels) in enumerate(val_dataloader):
            outputs = model(val_inputs)
            loss = criterion(outputs, val_labels)
            val_loss += loss.item()
    wandb.log({"val_loss": val_loss})
    print(f"Validation loss: {val_loss / len(val_dataloader)}")

    # Save the model if it has the best validation loss so far
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")

# Save the trained model
torch.save(model.state_dict(), "final_model.pth")

# Convert to TorchScript using tracing
# example_input = torch.randn(1, 10)
# traced_script_module = torch.jit.trace(model, example_input)
# traced_script_module.save("simple_model.pt")