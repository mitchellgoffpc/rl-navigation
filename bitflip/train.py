import time
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass

from common.metrics import Metrics, MetricType
from common.replay import ReplayBuffer
from common.config import BaseConfig
from bitflip.models import BitflipAgent
from bitflip.environment import BitflipEnvironment

@dataclass
class BitflipConfig(BaseConfig):
  lr = 0.0003
  batch_size = 16
  bit_length = 4
  hidden_size = 128
  batch_size = 512
  num_episodes = 20000
  max_episode_length = 12
  replay_buffer_size = 1000
  report_interval = 100


def train(config):
  metric_types = {"Wins": MetricType.SUM, "Episode Length": MetricType.MEAN, "Epsilon": MetricType.MEAN}
  metrics = Metrics(config.report_interval, metric_types)
  replay = ReplayBuffer(config.replay_buffer_size)
  env = BitflipEnvironment(config.bit_length)
  agent = BitflipAgent(config.hidden_size, config.bit_length)
  optimizer = torch.optim.Adam(agent.parameters(), lr=config.lr)

  for episode_counter in range(1, config.num_episodes + 1):
    epsilon = max(0.00, 1. - 3 * float(episode_counter) / config.num_episodes)

    # Roll out an episode
    state, goal = env.reset()
    for step in range(config.max_episode_length):
      with torch.no_grad():
        distances = agent(state[None], goal[None])

      # Epsilon-greedy exploration strategy
      if np.random.uniform() > epsilon:
        action = torch.argmin(distances).numpy()
      else:
        action = np.random.randint(0, config.bit_length)

      next_state, done = env.step(action)
      replay.add_step(state, goal, action, next_state, done)
      state = next_state
      if done:
        break

    replay.commit()

    # Train the agent
    if replay.num_steps() > config.batch_size:
      states, goals, actions, next_states, finished, _ = (x.squeeze() for x in replay.sample(config.batch_size))

      with torch.no_grad():
        best_future_distances = torch.clip(agent(next_states, goals).min(dim=-1).values * ~finished, 0, config.bit_length)
      distances = agent(states, goals)[torch.arange(len(actions)), actions]
      loss = F.smooth_l1_loss(distances, best_future_distances + 1)
      loss.backward()

      optimizer.step()
      optimizer.zero_grad()

    metrics.add({"Wins": done, "Episode Length": step, "Epsilon": epsilon})


# ENTRY POINT

if __name__ == '__main__':
  train(BitflipConfig.from_cli())
