import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from zelda.helpers import get_device
from zelda.datasets import ImitationDataset
from zelda.vqmodels import VQModel

NUM_EPOCHS = 10
BATCH_SIZE = 1
LEARNING_RATE = 3e-4
NUM_ACTIONS = 4

device = get_device()
model = VQModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_dataset = ImitationDataset('random', subsample=0.1)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(NUM_EPOCHS):
  model.train()
  train_loss, correct, total = 0, 0, 0
  for state, _, _, _ in (pbar := tqdm(train_loader, leave=False)):
    state = state.float().to(device)
    optimizer.zero_grad()
    outputs = model(state)
    print(outputs)
    loss = F.cross_entropy(outputs, action)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    total += action.size(0)
    correct += (outputs.argmax(dim=1) == action).sum().item()
    pbar.set_description(f"[TRAIN] Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {loss.item():.3f}")

    print(f"[TRAIN] Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {train_loss/len(train_loader):.3f}")
