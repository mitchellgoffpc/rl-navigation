import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.replay import ReplayBuffer
from maze.environment import MazeEnv

NUM_EPISODES = 10000
NUM_STEPS = 20
NUM_TRAINING_STEPS = 10
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
  distance_model = MLP(math.prod(env.observation_space.shape), 1)
  policy_model = MLP(math.prod(env.observation_space.shape), env.action_space.n)
  distance_optimizer = torch.optim.Adam(distance_model.parameters(), lr=3e-4)
  policy_optimizer = torch.optim.Adam(policy_model.parameters(), lr=3e-4)
  wins = 0
  total_loss = 0

  for episode in range(1, NUM_EPISODES + 1):

    # generate rollout
    distance_model.eval()
    state, _ = env.reset()
    goal = state.copy()
    goal[state == 2] = 1
    goal[state == 3] = 2
    for _ in range(NUM_STEPS):
      # action = random.randrange(env.action_space.n)
      with torch.no_grad():
        preds = policy_model(torch.as_tensor(state)[None], torch.as_tensor(goal)[None])
        action = torch.multinomial(F.softmax(preds, dim=-1), 1).item()
      next_state, reward, done, _, _ = env.step(action)
      replay.add_step(state, action, reward, next_state, done)
      state = next_state
      if done:
        wins += 1
        break

    replay.commit()

    # train on samples from replay buffer
    distance_model.train()
    for _ in range(NUM_TRAINING_STEPS):
      states, actions, _, next_states, _, mask = replay.sample(BATCH_SIZE, NUM_STEPS)
      targets = np.random.randint(0, np.sum(mask, axis=-1), size=BATCH_SIZE)
      goals = states[np.arange(BATCH_SIZE), targets]

      preds = distance_model(torch.as_tensor(states[:, 0]), torch.as_tensor(goals))
      loss = F.smooth_l1_loss(preds, torch.as_tensor(targets)[:, None].float())
      total_loss += loss.item()
      distance_optimizer.zero_grad()
      loss.backward()
      distance_optimizer.step()

      advantages = torch.as_tensor(targets) - distance_model(torch.as_tensor(states[:, 0]), torch.as_tensor(goals)).squeeze()
      mask = advantages < advantages.mean()
      preds = policy_model(torch.as_tensor(states)[mask, 0], torch.as_tensor(goals)[mask])
      loss = F.cross_entropy(preds, torch.as_tensor(actions)[mask, 0])
      policy_optimizer.zero_grad()
      loss.backward()
      policy_optimizer.step()

      # advantages = distance_model(torch.as_tensor(states[:, 0]), torch.as_tensor(goals)) - distance_model(torch.as_tensor(next_states[:, 0]), torch.as_tensor(goals))
      # advantages = advantages.detach().squeeze()
      # advantages = advantages - advantages.mean()
      # preds = policy_model(torch.as_tensor(states[:, 0]), torch.as_tensor(goals))
      # preds = F.softmax(preds, dim=-1)
      # preds = preds[torch.arange(BATCH_SIZE), actions[:, 0]]
      # loss = -(preds * advantages).mean()
      # policy_optimizer.zero_grad()
      # loss.backward()
      # policy_optimizer.step()

      # if episode % 100 == 0:
      #   print(states[0, 0])
      #   print(next_states[0, 0])
      #   print(goals[0])
      #   print(actions[0, 0], advantages[0], preds[0], targets[0])
      #   print()


    # print metrics
    if episode % 100 == 0:
      evaluated_wins = 0
      evaluated_steps = 0
      starting_distances = []
      episode_wins = []

      distance_model.eval()
      for _ in range(100):
        state, info = env.reset()
        starting_distances.append(np.abs(np.argwhere(state == 2) - np.argwhere(state == 3)).sum())
        goal = state.copy()
        goal[state == 2] = 1
        goal[state == 3] = 2
        for step in range(NUM_STEPS):
          # distances = []
          # for action in MazeEnv.MOVES:
          #   next_state, _ = env.next_obs(action)
          #   distances.append(distance_model(torch.as_tensor(next_state)[None], torch.as_tensor(goal)[None]).squeeze().item())
          # norm_distances = np.array(distances) - np.mean(distances)
          # action = torch.multinomial(F.softmax(torch.tensor(-norm_distances), dim=-1), 1).item()
          preds = policy_model(torch.as_tensor(state)[None], torch.as_tensor(goal)[None])
          action = torch.multinomial(F.softmax(preds, dim=-1), 1).item()
          state, _, done, _, _ = env.step(action)
          if done:
            evaluated_wins += 1
            break

        episode_wins.append(step < NUM_STEPS - 1)
        evaluated_steps += step + 1

      import matplotlib.pyplot as plt
      plt.figure(figsize=(12, 6))
      unique_distances = np.unique(starting_distances)
      avg_wins = [np.mean(np.array(episode_wins)[np.array(starting_distances) == d]) for d in unique_distances]
      plt.bar(unique_distances, avg_wins)
      plt.xlabel('Starting Distance')
      plt.ylabel('Number of Wins')
      plt.title('Number of Wins for each Starting Distance')
      plt.show()

      print(f"Episode: {episode}, wins: {wins}, average loss: {total_loss / (NUM_TRAINING_STEPS * 10):.4f}, eval wins: {evaluated_wins}, eval episode length: {evaluated_steps / 100:.2f}")
      wins = 0
      total_loss = 0



if __name__ == '__main__':
  train()
