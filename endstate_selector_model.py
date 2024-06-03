import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import wandb
import numpy as np
import matplotlib.pyplot as plt

# Define a simple model
class FCNStateSelector(nn.Module):
    def __init__(self):
        super(FCNStateSelector, self).__init__()
        self.fc1 = nn.Linear(10, 1)
        self.fc2 = nn.Linear(10, (18*2+1)*(19*2+1))

    def forward(self, x):
        x = self.fc1(x)
        x = F.softmax(x)
        return x
    
# Could conv along each x,y,z axis, along each waypoint
class CNNStateSelector(nn.Module):
    def __init__(self):
        super(CNNStateSelector, self).__init__()
        self.conv1 = nn.Conv1d(1, 20, 5)
        self.conv2 = nn.Conv1d(20, 50, 5)
        self.fc1 = nn.Linear(50, 10)
        self.fc2 = nn.Linear(10, (18*2+1)*(19*2+1))
        # self.fc2 = nn.Linear(10, 1) #output is a continuous value but since the outputs are discrete, we need to use a softmax layer
        # should train like a classification problem since I have that for the outputs and anything more granular wouldn't make a signficant difference in execution time
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = x.view(-1, 50)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.Softmax(x)
        return x

# Input sequence may be so short that LSTM is not necessary
class LSTMStateSelector(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMStateSelector, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
def train_data_prep(X, y): # was thinking of subtracting to make first waypoint 0,0,0 but then I'd have to do that at inference time too probably
    X_train_valid = np.load('X_train_valid.npy') # (N, 4, 3)
    y_train_valid = np.load('y_train_valid.npy') # (N, 4) v_mag, v_dir, a_mag, a_dir


def test_data_prep(X, y):
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')

# Initialize and train your model (example)
model = FCNStateSelector()
criterion = nn.MSELoss()
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
