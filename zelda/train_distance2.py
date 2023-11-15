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
    current_pos = torch.stack(info['screen_pos'], -1)
    goal_pos = torch.stack(info['goal_screen_pos'], -1)
    distance = (current_pos - goal_pos).abs().sum(-1)
    return distance / 10  # NOTE: the actual distance moved in a single action can vary, 10 is a rough approximation

def get_true_actions(info):
    current_pos = torch.stack(info['screen_pos'], -1)
    goal_pos = torch.stack(info['goal_screen_pos'], -1)
    actions = torch.zeros((len(current_pos), 4), dtype=torch.bool)
    actions[goal_pos[:,1] < current_pos[:,1], 0] = 1  # UP
    actions[goal_pos[:,1] > current_pos[:,1], 1] = 1  # DOWN
    actions[goal_pos[:,0] < current_pos[:,0], 2] = 1  # LEFT
    actions[goal_pos[:,0] > current_pos[:,0], 3] = 1  # RIGHT
    actions[(goal_pos == current_pos).all(-1)] = 1  # NO-OP
    return actions


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


# TRAIN MODELS

device = (
    torch.device('cuda') if torch.cuda.is_available() else
    torch.device('mps') if torch.backends.mps.is_available() else
    torch.device('cpu'))

distance_model = ZeldaAgent(2).to(device)
policy_model = ZeldaAgent(4).to(device)
distance_optim = torch.optim.Adam(distance_model.parameters(), lr=LEARNING_RATE)
policy_optim = torch.optim.Adam(policy_model.parameters(), lr=LEARNING_RATE)

train_dataset = EdgeDataset(train_edges)
test_dataset = EdgeDataset(test_edges)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

for epoch in range(NUM_EPOCHS):

    # TRAIN

    distance_model.train()
    policy_model.train()
    dist_err, policy_acc = [], []
    true_dist, pred_dist = [], []

    for state, goal_state, action, info in (pbar := tqdm(train_loader, leave=False)):
        state, goal_state, action, distance = state.to(device), goal_state.to(device), action.to(device), info['distance'].float().to(device)

        distance_optim.zero_grad()
        distance_preds = distance_model(state, goal_state)
        distance_loss = F.gaussian_nll_loss(distance_preds[:,0], distance, distance_preds[:,1].exp())
        distance_loss.backward()
        distance_optim.step()

        policy_optim.zero_grad()
        action_preds = policy_model(state, goal_state)
        policy_loss = F.cross_entropy(action_preds, action)
        policy_loss.backward()
        policy_optim.step()

        dist_err.append(F.mse_loss(distance_preds[:,0], distance).item())
        true_actions = get_true_actions(info)
        policy_acc.append(((action_preds.softmax(-1).cpu() * true_actions).sum(-1) / true_actions.sum(-1)).mean().item())
        true_dist.extend(get_true_distance(info).cpu().numpy().tolist())
        pred_dist.extend(distance_preds[:,0].cpu().detach().numpy().tolist())
        pbar.set_description(f"[TRAIN] Epoch {epoch+1}/{NUM_EPOCHS} | Distance MSE: {np.mean(dist_err):.2f} | Policy Accuracy: {np.mean(policy_acc):.3f}")

    print(f"[TRAIN] Epoch {epoch+1}/{NUM_EPOCHS} | Distance MSE: {np.mean(dist_err):.2f} | Policy Accuracy: {np.mean(policy_acc):.3f}")

    plt.scatter(true_dist, pred_dist)
    plt.xlabel('True Distance')
    plt.ylabel('Predicted Distance')
    plt.title('Training: True vs Predicted Distance')
    plt.show()

    # TEST

    distance_model.eval()
    policy_model.eval()
    dist_err, policy_acc = [], []
    true_dist, pred_dist = [], []

    for state, goal_state, _, info in (pbar := tqdm(test_loader, leave=False)):
        state, goal_state, distance = state.to(device), goal_state.to(device), info['distance'].float().to(device)
        with torch.no_grad():
            distance_preds = distance_model(state, goal_state)
            action_preds = policy_model(state, goal_state)
        distance_loss = F.gaussian_nll_loss(distance_preds[:,0], distance, distance_preds[:,1].exp())

        dist_err.append(F.mse_loss(distance_preds[:,0], distance).item())
        true_actions = get_true_actions(info)
        policy_acc.append(((action_preds.softmax(-1).cpu() * true_actions).sum(-1) / true_actions.sum(-1)).mean().item())
        true_dist.extend(get_true_distance(info).cpu().numpy().tolist())
        pred_dist.extend(distance_preds[:,0].cpu().detach().numpy().tolist())
        pbar.set_description(f"[TEST]  Epoch {epoch+1}/{NUM_EPOCHS} | Distance MSE: {np.mean(dist_err):.2f} | Policy Accuracy: {np.mean(policy_acc):.3f}")

    print(f"[TEST]  Epoch {epoch+1}/{NUM_EPOCHS} | Distance MSE: {np.mean(dist_err):.2f} | Policy Accuracy: {np.mean(policy_acc):.3f}\n")

    plt.scatter(true_dist, pred_dist)
    plt.xlabel('True Distance')
    plt.ylabel('Predicted Distance')
    plt.title('Testing: True vs Predicted Distance')
    plt.show()
