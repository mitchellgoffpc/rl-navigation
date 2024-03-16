import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
  def __init__(self, input_size, output_size):
      super().__init__()
      self.embed = nn.Embedding(4, 16)
      self.conv1 = nn.Conv2d(32, 32, 3, padding=1)
      self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
      self.fc1 = nn.Linear(input_size * 64, 128)
      self.fc2 = nn.Linear(128, output_size)

  def forward(self, state, goal):
    x = torch.stack([state, goal], dim=1).long()
    x = self.embed(x)
    x = x.permute(0, 1, 4, 2, 3).reshape(x.size(0), -1, x.size(2), x.size(3))
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = x.flatten(start_dim=1)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x