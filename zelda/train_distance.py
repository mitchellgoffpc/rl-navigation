import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader
from common.helpers import get_device, save_checkpoint
from zelda.models import ResNet
from zelda.environment import ZeldaEnvironment
from zelda.helpers import generate_rollout, generate_goals, evaluate_model


NUM_EPISODES = 16
NUM_STEPS = 20
NUM_REPEATS = 8
BATCH_SIZE = 32
LEARNING_RATE = 3e-4

class EdgeDataset(Dataset):
  def __init__(self, data):
    self.data = data

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]

def get_true_distance(info):
  current_pos = torch.stack(info['pos'][:2], -1)
  goal_pos = torch.stack(info['goal_pos'][:2], -1)
  distance = (current_pos - goal_pos).abs().sum(-1)
  return distance / 10  # NOTE: the actual distance moved in a single action can vary, 10 is a rough approximation

def get_true_actions(info):
  current_pos = torch.stack(info['pos'][:2], -1)
  goal_pos = torch.stack(info['goal_pos'][:2], -1)
  actions = torch.zeros((len(current_pos), 4), dtype=torch.bool)
  actions[goal_pos[:,1] < current_pos[:,1], 0] = 1  # UP
  actions[goal_pos[:,1] > current_pos[:,1], 1] = 1  # DOWN
  actions[goal_pos[:,0] < current_pos[:,0], 2] = 1  # LEFT
  actions[goal_pos[:,0] > current_pos[:,0], 3] = 1  # RIGHT
  actions[(goal_pos == current_pos).all(-1)] = 1  # NO-OP
  return actions


# GENERATE DATA

def generate_episode(env, policy_model, goal):
  episode = generate_rollout(env, NUM_STEPS)

  edges = []
  for i in range(len(episode)):
    for j in range(i+1, len(episode)):
      state, action, info = episode[i]
      goal_state, _, goal_info = episode[j]
      info = {'distance': j - i, 'pos': info['pos'], 'goal_pos': goal_info['pos']}
      edges.append((state, goal_state, action, info))

  return edges

def get_rollout_data(env, distance_model, policy_model, prev_edges, num_episodes):
  if False: # prev_edges is not None:
    obs, info = env.reset()
    edges = random.sample(prev_edges, k=num_episodes * 10)
    candidate_goals = np.stack([x[1] for x in edges], axis=0)
    # obs = obs[None].repeat(len(edges), 0)
    # distances = distance_model(obs, candidate_goals)
    positions = np.array([edge[-1]['goal_screen_pos'] for edge in edges], dtype=int)
    distances = np.abs(np.array(info['screen_pos'], dtype=int) - positions).sum(axis=1, keepdims=True)

    # goal_indices = sorted(range(len(edges)), key=lambda i: distances[i,0].item(), reverse=True)
    goal_indices = list(range(len(edges)))
    random.shuffle(goal_indices)
    goals = [candidate_goals[i] for i in goal_indices[:num_episodes]]
    goal_distances = [distances[i,0] for i in goal_indices[:num_episodes]]
  else:
    goals = [None] * num_episodes

  edges = []
  for i in trange(num_episodes, desc='generating episodes'):
    edges.extend(generate_episode(env, policy_model, goals[i]))
  return edges


# TRAIN MODELS

def train_distance_model(device, model, train_edges, test_edges):
  optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

  train_dataset = EdgeDataset(train_edges)
  test_dataset = EdgeDataset(test_edges)
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

  model.train()
  dist_err, true_dist, pred_dist = [], [], []
  for state, goal_state, action, info in (pbar := tqdm(train_loader)):
    optim.zero_grad()
    distance = info['distance'].float().to(device)
    distance_preds = model(state, goal_state)
    loss = F.gaussian_nll_loss(distance_preds[:,0], distance, distance_preds[:,1].exp())
    loss.backward()
    optim.step()

    dist_err.append(F.mse_loss(distance_preds[:,0], distance).item())
    true_dist.extend(get_true_distance(info).cpu().numpy().tolist())
    pred_dist.extend(distance_preds[:,0].cpu().detach().numpy().tolist())
    pbar.set_description(f"[TRAIN] Distance MSE: {np.mean(dist_err):.2f}")

  model.eval()
  dist_err, true_dist, pred_dist = [], [], []
  for state, goal_state, _, info in (pbar := tqdm(test_loader)):
    with torch.no_grad():
      distance_preds = model(state, goal_state)

    distance = info['distance'].float().to(device)
    dist_err.append(F.mse_loss(distance_preds[:,0], distance).item())
    true_dist.extend(get_true_distance(info).cpu().numpy().tolist())
    pred_dist.extend(distance_preds[:,0].cpu().detach().numpy().tolist())
    pbar.set_description(f"[TEST]  Distance MSE: {np.mean(dist_err):.2f}")


def train_policy_model(device, model, train_edges, test_edges):
  optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

  train_dataset = EdgeDataset(train_edges)
  test_dataset = EdgeDataset(test_edges)
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

  model.train()
  policy_acc = []
  for state, goal_state, action, info in (pbar := tqdm(train_loader)):
    optim.zero_grad()
    action_preds = model(state, goal_state)
    loss = F.cross_entropy(action_preds, action.to(device))
    loss.backward()
    optim.step()

    true_actions = get_true_actions(info)
    policy_acc.append((action_preds.softmax(-1).cpu() * true_actions).sum(-1).mean().item())
    pbar.set_description(f"[TRAIN] Policy Accuracy: {np.mean(policy_acc):.3f}")

  policy_model.eval()
  policy_acc = []
  for state, goal_state, _, info in (pbar := tqdm(test_loader)):
    with torch.no_grad():
      action_preds = policy_model(state, goal_state)

    true_actions = get_true_actions(info)
    policy_acc.append((action_preds.softmax(-1).cpu() * true_actions).sum(-1).mean().item())
    pbar.set_description(f"[TEST]  Policy Accuracy: {np.mean(policy_acc):.3f}")


def filter_edges(distance_model, edges, num_edges):
  dataset = EdgeDataset(edges)
  loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

  edge_scores = []
  for state, goal_state, _, info in tqdm(loader, desc="[FILTERING]"):
    with torch.no_grad():
      distance_preds = distance_model(state, goal_state)
    means, stds = distance_preds[:,0].cpu(), distance_preds[:,1].exp().cpu()
    # scores = (info['distance'] - means) / stds
    scores = info['distance'] / torch.clamp(means, 1, np.inf)
    edge_scores.extend(scores.numpy().tolist())

  best_edge_idxs = np.argsort(edge_scores)[:num_edges]
  return [edges[i] for i in best_edge_idxs]


# TRAINING LOOP

if __name__ == "__main__":
  device = get_device()
  env = ZeldaEnvironment()
  distance_model = ResNet(2).to(device)
  policy_model = ResNet(4).to(device)
  train_edges, test_edges = None, None

  for i in range(3):
    # collect data
    test_edges = get_rollout_data(env, distance_model, policy_model, train_edges, NUM_EPISODES)  # test rollouts use train_edges as well
    train_edges = get_rollout_data(env, distance_model, policy_model, train_edges, NUM_EPISODES * 2)

    random.shuffle(test_edges)
    test_edges = test_edges[:len(test_edges) // 10]

    # reset the distance model
    distance_model = ResNet(2).to(device)

    # train distance and policy models
    train_distance_model(device, distance_model, train_edges[:len(train_edges) // 2], test_edges)
    train_edges = filter_edges(distance_model, train_edges, len(train_edges) // 4)
    train_policy_model(device, policy_model, train_edges, test_edges)

    def policy(obs, goal):
      action_probs = policy_model(obs, goal).softmax(-1)
      action = torch.multinomial(action_probs, 1).cpu().numpy().squeeze().item()
      return action

    goal_states, goal_positions = generate_goals(env, num_goals=16, num_steps=NUM_STEPS)
    eval_results = evaluate_model(env, policy, goal_states, goal_positions, NUM_STEPS)
    print(eval_results.mean())

    # save models
    save_checkpoint(distance_model, 'distance.ckpt')
    save_checkpoint(policy_model, 'policy.ckpt')
