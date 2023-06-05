import torch
import torch.nn as nn
import torch.nn.functional as F

class BitflipAgent(nn.Module):
  def __init__(self, hidden_size, bit_length):
    super().__init__()
    self.fc1 = nn.Linear(bit_length * 2, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
    self.fc3 = nn.Linear(hidden_size // 2, bit_length)

  def forward(self, inputs, goals):
    input = torch.cat((inputs, goals), dim=1).float()
    x = F.relu(self.fc1(input))
    x = F.relu(self.fc2(x))
    return self.fc3(x)
