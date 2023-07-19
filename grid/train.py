import time
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass

from common.metrics import Metrics, MetricType
from common.graph import ReplayGraph
from common.config import BaseTrainingConfig
from grid.models import GridAgent, NUM_ACTIONS
from grid.environment import GridEnvironment

@dataclass
class GridConfig(BaseTrainingConfig):
  lr: float = 0.0003
  grid_size: int = 4
  batch_size: int = 512
  hidden_size: int = 128
  num_episodes: int = 1000
  max_episode_length: int = 8
  replay_buffer_size: int = 1000
  goal_replacement_prob: float = 0.5
  num_train_steps: int = 4
  report_interval: int = 100
  device_name: str = 'cpu'


def train(config):
  metric_types = {"Wins": MetricType.SUM, "Episode Length": MetricType.MEAN, "Epsilon": MetricType.MEAN}
  metrics = Metrics(config.report_interval, metric_types)
  replay = ReplayGraph(config.replay_buffer_size)
  env = GridEnvironment(config.grid_size, config.grid_size)
  agent = GridAgent(config.grid_size * config.grid_size, config.hidden_size).to(config.device)
  optimizer = torch.optim.Adam(agent.parameters(), lr=config.lr)

  for episode_counter in range(1, config.num_episodes + 1):
    epsilon = max(0.00, 1. - 2 * float(episode_counter) / config.num_episodes)

    # Roll out an episode
    state, goal = env.reset()
    for step in range(1, config.max_episode_length + 1):
      # Epsilon-greedy exploration strategy
      if np.random.uniform() > epsilon:
        with torch.no_grad():
          distances = agent(state[None], goal[None]).cpu()
          action = torch.argmin(distances).numpy()
      else:
        action = np.random.randint(0, NUM_ACTIONS)

      next_state, _, done, info = env.step(action)
      replay.add_step(state, goal, action, next_state, done, info['pos'], info['goal_pos'])
      state = next_state
      if done:
        break

    replay.commit()
    if episode_counter % 10 == 0:
      replay.compile(lambda x, y: np.inf, key=lambda s, *_: hash(s.tobytes()))

    # Train the agent
    if len(replay.episodes) > 50:
      for _ in range(config.num_train_steps):
        states, goals, actions, targets = replay.sample(config.batch_size)
        distances = agent(states, goals)[torch.arange(len(actions)), actions]
        loss = F.smooth_l1_loss(distances, torch.clip(torch.as_tensor(targets), 0, config.max_episode_length))
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    metrics.add({"Wins": done, "Episode Length": step, "Epsilon": epsilon})


# ENTRY POINT

if __name__ == '__main__':
  train(GridConfig.from_cli())
