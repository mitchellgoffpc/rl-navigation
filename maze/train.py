import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.replay import ReplayBuffer
from maze.environment import MazeEnv

NUM_EPISODES = 1000
NUM_STEPS = 100
NUM_TRAINING_STEPS = 1
BATCH_SIZE = 64

class MLP(nn.Module):
  def __init__(self, input_size, output_size):
      super().__init__()
      self.embed = nn.Embedding(4, 4)
      self.fc1 = nn.Linear(input_size * 2 * 4, 128)
      self.fc2 = nn.Linear(128, 64)
      self.fc3 = nn.Linear(64, output_size)

  def forward(self, state, goal):
    x = torch.cat([state, goal], dim=-1).long()
    x = self.embed(x).flatten(start_dim=1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x


def train():
  env = MazeEnv(5, 5)
  replay = ReplayBuffer(1000)
  model = MLP(math.prod(env.observation_space.shape), env.action_space.n)
  optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
  wins = 0
  total_loss = 0

  for episode in range(1, NUM_EPISODES + 1):

    # generate rollout
    model.eval()
    state, _ = env.reset()
    for _ in range(NUM_STEPS):
      action = random.randrange(env.action_space.n)
      next_state, reward, done, _, _ = env.step(action)
      replay.add_step(state, action, reward, next_state, done)
      state = next_state
      if done:
        wins += 1
        break

    replay.commit()

    # train on samples from replay buffer
    if len(replay) < 10: continue

    model.train()
    for _ in range(NUM_TRAINING_STEPS):
      states, actions, _, next_states, _, mask = replay.sample(BATCH_SIZE, NUM_STEPS)
      targets = np.random.randint(0, np.sum(mask, axis=-1), size=BATCH_SIZE)
      goals = next_states[np.arange(BATCH_SIZE), targets]

      preds = model(torch.as_tensor(states[:, 0]), torch.as_tensor(goals))
      preds = preds[torch.arange(len(actions)), actions[:, 0]]
      loss = F.mse_loss(preds, torch.as_tensor(targets) / 100)
      total_loss += loss.item()
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

    # print metrics
    if episode % 10 == 0:
      evaluated_wins = 0
      evaluated_steps = 0
      model.eval()
      for _ in range(10):
        state, _ = env.reset()
        goal = state.copy()
        goal[state == 2] = 1
        goal[state == 3] = 2
        for step in range(NUM_STEPS):
          distances = model(torch.as_tensor(state)[None], torch.as_tensor(goal)[None]).squeeze()
          action = torch.argmin(distances).item()
          state, _, done, _, _ = env.step(action)
          if done:
            evaluated_wins += 1
            break
        evaluated_steps += step + 1

      print(f"Episode: {episode}, wins: {wins}, average loss: {total_loss / (NUM_TRAINING_STEPS * 10):.4f}, eval wins: {evaluated_wins}, eval episode length: {evaluated_steps / 10:.2f}")
      wins = 0
      total_loss = 0



if __name__ == '__main__':
  train()
