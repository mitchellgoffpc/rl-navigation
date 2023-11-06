import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from zelda.models import ZeldaAgent
from zelda.datasets import ImitationDataset

NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 3e-4

device = (
    torch.device('cuda') if torch.cuda.is_available() else
    torch.device('mps') if torch.backends.mps.is_available() else
    torch.device('cpu'))

model = ZeldaAgent().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_dataset = ImitationDataset('random', subsample=0.1, max_goal_dist=2)
test_dataset = ImitationDataset('expert', max_episode_len=10, max_goal_dist=1)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss, correct, total = 0, 0, 0
    for state, goal_state, action, _ in (pbar := tqdm(train_loader, leave=False)):
        state, goal_state, action = state.to(device), goal_state.to(device), action.to(device)
        optimizer.zero_grad()
        outputs = model(state, goal_state)
        loss = F.cross_entropy(outputs, action)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total += action.size(0)
        correct += (outputs.argmax(dim=1) == action).sum().item()
        pbar.set_description(f"[TRAIN] Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {loss.item():.3f} | Accuracy: {100*correct/total:.2f}%")

    print(f"[TRAIN] Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {train_loss/len(train_loader):.3f} | Accuracy: {100.*correct/total:.2f}%")

    model.eval()
    test_loss, correct, total = 0, 0, 0
    for state, goal_state, action, _ in (pbar := tqdm(test_loader, leave=False)):
        state, goal_state, action = state.to(device), goal_state.to(device), action.to(device)
        with torch.no_grad():
            outputs = model(state, goal_state)
        loss = F.cross_entropy(outputs, action)

        test_loss += loss.item()
        total += action.size(0)
        correct += (outputs.argmax(dim=1) == action).sum().item()
        pbar.set_description(f"[TEST] Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {loss.item():.3f} | Accuracy: {100*correct/total:.2f}")

    print(f"[TEST] Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {test_loss/len(test_loader):.3f} Accuracy: {100.*correct/total:.2f}%")
