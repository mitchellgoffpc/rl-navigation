import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
from zelda.models import ZeldaAgent
from zelda.environment import ZeldaEnvironment

NUM_EPISODES = 20
NUM_STEPS = 20
NUM_REPEATS = 8
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 3e-4

class EdgeDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def generate_episode(env):
    episode = []
    obs, info = env.reset()
    actions = [env.UP, env.DOWN, env.LEFT, env.RIGHT]

    for j in range(NUM_STEPS):
        action = np.random.randint(len(actions))
        episode.append((obs, action, info['map_pos'], info['screen_pos']))
        for _ in range(NUM_REPEATS):
            obs, info = env.step(actions[action])

    edges = []
    for i in range(len(episode)):
        for j in range(i+1, len(episode)):
            state, action, map_pos, screen_pos = episode[i]
            goal_state, _, goal_map_pos, goal_screen_pos = episode[j]
            info = {
                'distance': j - i,
                'map_pos': map_pos, 'screen_pos': screen_pos,
                'goal_map_pos': goal_map_pos, 'goal_screen_pos': goal_screen_pos}
            edges.append((state, goal_state, action, info))

    return edges

def get_true_distance(info):
    map_pos, screen_pos = info['map_pos'], info['screen_pos']
    goal_map_pos, goal_screen_pos = info['goal_map_pos'], info['goal_screen_pos']
    distance = (screen_pos[0] - goal_screen_pos[0]).abs() + (screen_pos[1] - goal_screen_pos[1]).abs()
    return distance / 10


# GENERATE DATA

env = ZeldaEnvironment()
train_edges, test_edges = [], []

for _ in trange(NUM_EPISODES, desc='generating train episodes'):
    train_edges.extend(generate_episode(env))
for _ in trange(NUM_EPISODES, desc='generating test episodes'):
    test_edges.extend(generate_episode(env))

import random
random.shuffle(test_edges)
test_edges = test_edges[:len(test_edges) // 10]

print(f"Collected {len(train_edges)} train edges and {len(test_edges)} test edges")


# TRAIN A MODEL

device = (
    torch.device('cuda') if torch.cuda.is_available() else
    torch.device('mps') if torch.backends.mps.is_available() else
    torch.device('cpu'))

model = ZeldaAgent(2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_dataset = EdgeDataset(train_edges)
test_dataset = EdgeDataset(test_edges)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

for epoch in range(NUM_EPOCHS):

    # TRAIN

    model.train()
    train_loss, err, total = 0, 0, 0
    true_dist, pred_dist = [], []

    for state, goal_state, _, info in (pbar := tqdm(train_loader, leave=False)):
        state, goal_state, distance = state.to(device), goal_state.to(device), info['distance'].float().to(device)
        optimizer.zero_grad()
        outputs = model(state, goal_state)
        loss = F.gaussian_nll_loss(outputs[:,0], distance, outputs[:,1].exp())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total += 1
        err += F.mse_loss(outputs[:,0], distance).item()
        true_dist.extend(get_true_distance(info).cpu().numpy().tolist())
        pred_dist.extend(outputs[:,0].cpu().detach().numpy().tolist())
        pbar.set_description(f"[TRAIN] Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {loss.item():.3f} | MSE: {err/total:.2f}")

    print(f"[TRAIN] Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {train_loss/len(train_loader):.3f} | MSE: {err/total:.2f}")

    plt.scatter(true_dist, pred_dist)
    plt.xlabel('True Distance')
    plt.ylabel('Predicted Distance')
    plt.title('Training: True vs Predicted Distance')
    plt.show()

    # TEST

    model.eval()
    test_loss, err, total = 0, 0, 0
    true_dist, pred_dist = [], []

    for state, goal_state, _, info in (pbar := tqdm(test_loader, leave=False)):
        state, goal_state, distance = state.to(device), goal_state.to(device), info['distance'].float().to(device)
        with torch.no_grad():
            outputs = model(state, goal_state)
        loss = F.gaussian_nll_loss(outputs[:,0], distance, outputs[:,1].exp())

        test_loss += loss.item()
        total += 1
        err += F.mse_loss(outputs[:,0], distance).item()
        true_dist.extend(get_true_distance(info).cpu().numpy().tolist())
        pred_dist.extend(outputs[:,0].cpu().detach().numpy().tolist())
        pbar.set_description(f"[TEST]  Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {loss.item():.3f} | MSE: {err/total:.2f}")

    print(f"[TEST]  Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {test_loss/len(test_loader):.3f} | MSE: {err/total:.2f}\n")

    plt.scatter(true_dist, pred_dist)
    plt.xlabel('True Distance')
    plt.ylabel('Predicted Distance')
    plt.title('Testing: True vs Predicted Distance')
    plt.show()
