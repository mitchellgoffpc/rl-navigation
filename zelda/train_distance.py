import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm, trange
from zelda.models import ZeldaAgent
from zelda.environment import ZeldaEnvironment

NUM_EPISODES = 20
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

def generate_episode(env, policy_model, goal):
    episode = []
    obs, info = env.reset()
    actions = [env.UP, env.DOWN, env.LEFT, env.RIGHT]

    # Generate rollouts
    for j in range(NUM_STEPS):
        if goal is not None:
            with torch.no_grad():
                torch_obs = torch.as_tensor(obs[None])
                torch_goal = torch.as_tensor(goal[None])
                action_probs = policy_model(torch_obs, torch_goal).softmax(-1)
                action = torch.multinomial(action_probs, 1).cpu().numpy().item()
        else:
            action = np.random.randint(len(actions))

        episode.append((obs, action, info['map_pos'], info['screen_pos']))
        for _ in range(NUM_REPEATS):
            obs, info = env.step(actions[action])

    # Generate edges
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

def get_rollout_data(distance_model, policy_model, prev_edges, num_episodes):
    env = ZeldaEnvironment()
    obs, _ = env.reset()
    if prev_edges is not None:
      edges = random.sample(prev_edges, k=num_episodes * 10)
      obs = obs[None].repeat(len(edges), 0)
      candidate_goals = np.stack([x[1] for x in edges], axis=0)
      distances = distance_model(obs, candidate_goals)
      goal_indices = sorted(range(len(edges)), key=lambda i: distances[i,0].item(), reverse=True)
      goals = [candidate_goals[i] for i in goal_indices[:len(edges)]]
    else:
      goals = [None] * num_episodes

    return [edge for i in trange(num_episodes, desc='generating episodes')
                 for edge in generate_episode(env, policy_model, goals[i])]


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
    device = (
        torch.device('cuda') if torch.cuda.is_available() else
        torch.device('mps') if torch.backends.mps.is_available() else
        torch.device('cpu'))

    distance_model = ZeldaAgent(2).to(device)
    policy_model = ZeldaAgent(4).to(device)
    train_edges, test_edges = None, None

    for i in range(3):
        # collect data
        test_edges = get_rollout_data(distance_model, policy_model, train_edges, NUM_EPISODES)  # test rollouts use train_edges as well
        train_edges = get_rollout_data(distance_model, policy_model, train_edges, NUM_EPISODES * 2)

        random.shuffle(test_edges)
        test_edges = test_edges[:len(test_edges) // 10]

        # reset the distance model
        distance_model = ZeldaAgent(2).to(device)

        # train distance and policy models
        train_distance_model(device, distance_model, train_edges[:len(train_edges) // 2], test_edges)
        train_edges = filter_edges(distance_model, train_edges, len(train_edges) // 4)
        train_policy_model(device, policy_model, train_edges, test_edges)
        print("")

        # save models
        checkpoint_dir = Path(__file__).parent / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        torch.save(distance_model.state_dict(), checkpoint_dir / 'distance.ckpt')
        torch.save(policy_model.state_dict(), checkpoint_dir / 'policy.ckpt')
