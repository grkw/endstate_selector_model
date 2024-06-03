import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)

# Initialize and train your model (example)
model = SimpleModel()
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
