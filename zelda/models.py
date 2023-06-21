import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

INPUT_SIZE = (3, 224, 240)
NUM_ACTIONS = 8

class ZeldaAgent(nn.Module):
  def __init__(self, hidden_size):
    super().__init__()
    resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    self.backbone = torch.nn.Sequential(*list(resnet.children())[1:-1])
    self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.fc1 = nn.Linear(512, hidden_size)
    self.fc2 = nn.Linear(hidden_size, NUM_ACTIONS)

    # Load in the weights from the first resnet layer
    self.conv1.weight.data[:] = list(resnet.children())[0].weight.data.repeat(1, 2, 1, 1)

  def forward(self, state, goal):
    *b,h,w,c = state.shape
    state = torch.as_tensor(state).view(-1,h,w,c).permute(0,3,1,2).contiguous().float()
    goal = torch.as_tensor(goal).view(-1,h,w,c).permute(0,3,1,2).contiguous().float()
    x = torch.cat([state, goal], dim=1)
    x = self.conv1(x)
    x = self.backbone(x).flatten(start_dim=1)  # No relu for now
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x.view(*b,-1)
