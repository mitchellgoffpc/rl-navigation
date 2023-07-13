import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_ACTIONS = 4

class GridAgent(nn.Module):
  def __init__(self, input_size, hidden_size):
    super().__init__()
    self.fc1 = nn.Linear(input_size * 2, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
    self.fc3 = nn.Linear(hidden_size // 2, NUM_ACTIONS)

  def forward(self, inputs, goals):
    device = self.fc1.weight.device
    inputs, goals = (torch.as_tensor(x).flatten(start_dim=1).to(device) for x in (inputs, goals))
    input = torch.cat((inputs, goals), dim=-1).float()
    x = F.relu(self.fc1(input))
    x = F.relu(self.fc2(x))
    return self.fc3(x)
