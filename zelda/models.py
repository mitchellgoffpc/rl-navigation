import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

INPUT_SIZE = (240, 256, 3)
NUM_ACTIONS = 4

class ZeldaAgent(nn.Module):
  def __init__(self):
    super().__init__()
    resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    self.backbone = torch.nn.Sequential(*list(resnet.children())[1:-2])
    self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.conv2 = nn.Conv2d(512, 16, kernel_size=1, bias=False)
    self.fc1 = nn.Linear(16*8*8, 128)
    self.fc2 = nn.Linear(128, NUM_ACTIONS)

  def forward(self, state, goal):
    b,h,w,c = state.shape
    device = self.fc1.weight.device
    state = torch.as_tensor(state).permute(0,3,1,2).contiguous().float().to(device)
    goal = torch.as_tensor(goal).permute(0,3,1,2).contiguous().float().to(device)
    x = torch.cat([state, goal], dim=1) / 255
    x = self.conv1(x)
    x = self.backbone(x)
    x = self.conv2(x)
    x = x.flatten(start_dim=1)  # No relu for now
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x
