import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import trange, tqdm
from common.replay import ReplayBuffer
from maze.environment import MazeEnv

NUM_EPISODES = 100000
NUM_STEPS = 10
NUM_TRAINING_STEPS = 1
BATCH_SIZE = 128

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


def train():
  env = MazeEnv(7, 7)
  device = torch.device('cpu')
  distance_model = MLP(math.prod(env.observation_space.shape), 1)
  policy_model = MLP(math.prod(env.observation_space.shape), env.action_space.n).to(device)
  distance_optimizer = torch.optim.AdamW(distance_model.parameters(), lr=3e-4)
  policy_optimizer = torch.optim.AdamW(policy_model.parameters(), lr=3e-4)

  # collect dataset

  train_steps = []
  for _ in trange(NUM_EPISODES, desc="Generating train set"):
    state, _ = env.reset()
    for i in range(NUM_STEPS):
      correct_action = env.solve()[0]
      action = random.randrange(env.action_space.n)
      next_state, reward, done, _, _ = env.step(action)
      train_steps.append((state, correct_action))
      state = next_state
      if done:
        break

  val_steps = []
  for _ in trange(NUM_EPISODES, desc="Generating val set"):
    state, _ = env.reset()
    for _ in range(NUM_STEPS):
      correct_action = env.solve()[0]
      action = random.randrange(env.action_space.n)
      next_state, reward, done, _, _ = env.step(action)
      val_steps.append((state, correct_action))
      state = next_state
      if done:
        break


  # train policy

  for _ in range(5):

    # training
    train_loss = 0
    train_correct, train_total = 0, 0
    policy_model.train()
    random.shuffle(train_steps)
    for i in trange(0, len(train_steps), BATCH_SIZE, leave=False, desc="Train"):
      states, correct_actions = zip(*train_steps[i:i+BATCH_SIZE])
      states, correct_actions = np.array(states), np.array(correct_actions)
      preds = policy_model(torch.as_tensor(states), torch.as_tensor(states))
      loss = F.cross_entropy(preds, torch.as_tensor(correct_actions).to(device), reduction="none")
      policy_optimizer.zero_grad()
      loss.mean().backward()
      policy_optimizer.step()

      train_loss += loss.sum().item()
      train_correct += np.sum(np.argmax(preds.cpu().detach().numpy(), axis=-1) == correct_actions)
      train_total += len(states)

    # validation
    val_loss = 0
    val_correct, val_total = 0, 0
    policy_model.eval()
    with torch.no_grad():
      for i in trange(0, len(val_steps), BATCH_SIZE, leave=False, desc="Val"):
        states, correct_actions = zip(*val_steps[i:i+BATCH_SIZE])
        states, correct_actions = np.array(states), np.array(correct_actions)
        preds = policy_model(torch.as_tensor(states), torch.as_tensor(states))
        loss = F.cross_entropy(preds, torch.as_tensor(correct_actions).to(device), reduction="sum")
        val_loss += loss.item()
        val_correct += np.sum(np.argmax(preds.cpu().numpy(), axis=-1) == correct_actions)
        val_total += len(states)

    # print metrics
    torch.save(policy_model.state_dict(), Path(__file__).parent / 'policy.ckpt')
    print(f"POLICY: "
          f"Train accuracy: {train_correct / train_total * 100:.2f}%, Train loss: {train_loss / train_total:.4f}, "
          f"Val accuracy: {val_correct / val_total * 100:.2f}%, Val loss: {val_loss / val_total:.4f}")


  # train distance

  for _ in range(0):

    # training
    train_loss = 0
    train_total = 0
    distance_model.train()
    for i in trange(0, len(train_steps), BATCH_SIZE, leave=False, desc="Train"):
      indices = np.random.randint(0, len(train_steps), size=BATCH_SIZE)
      steps_diff = np.random.uniform(-20, 20, size=BATCH_SIZE).astype(int)
      goal_indices = np.clip(indices + steps_diff, 0, len(train_steps) - 1)
      states, _ = zip(*[train_steps[idx] for idx in indices])
      goals, _ = zip(*[train_steps[idx] for idx in goal_indices])
      states, goals = np.array(states), np.array(goals)
      steps_diff = np.abs(steps_diff)

      preds = distance_model(torch.as_tensor(states), torch.as_tensor(goals))
      loss = F.mse_loss(preds.squeeze(), torch.as_tensor(steps_diff, dtype=torch.float32).to(device), reduction="none")
      distance_optimizer.zero_grad()
      loss.mean().backward()
      distance_optimizer.step()
      train_loss += loss.sum().item()
      train_total += len(states)

    # validation
    val_loss = 0
    val_total = 0
    distance_model.eval()
    with torch.no_grad():
      for i in trange(0, len(val_steps), BATCH_SIZE, leave=False, desc="Val"):
        indices = np.arange(i, min(i + BATCH_SIZE, len(val_steps)))
        steps_diff = np.random.uniform(-20, 20, size=len(indices)).astype(int)
        goal_indices = np.clip(indices + steps_diff, 0, len(val_steps) - 1)
        states, _ = zip(*[val_steps[idx] for idx in indices])
        goals, _ = zip(*[val_steps[idx] for idx in goal_indices])
        states, goals = np.array(states), np.array(goals)
        steps_diff = np.abs(steps_diff)

        preds = distance_model(torch.as_tensor(states), torch.as_tensor(goals))
        loss = F.mse_loss(preds.squeeze(), torch.as_tensor(steps_diff, dtype=torch.float32).to(device), reduction="sum")
        val_loss += loss.item()
        val_total += len(states)

    # print metrics
    torch.save(distance_model.state_dict(), Path(__file__).parent / 'distance.ckpt')
    print(f"DISTANCE: "
          f"Train loss: {train_loss / train_total:.4f}, "
          f"Val loss: {val_loss / val_total:.4f}")


if __name__ == '__main__':
  train()
