import time
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass

from common.metrics import Metrics, MetricType
from common.replay import ReplayBuffer
from common.config import BaseConfig
from zelda.models import ZeldaAgent, NUM_ACTIONS
from zelda.environment import ZeldaEnvironment

@dataclass
class ZeldaConfig(BaseConfig):
  lr = 0.0003
  batch_size = 16
  hidden_size = 128
  batch_size = 32
  num_episodes = 200
  max_episode_length = 10
  replay_buffer_size = 1000
  report_interval = 10


def train(config):
  metric_types = {"Wins": MetricType.SUM, "Episode Length": MetricType.MEAN, "Epsilon": MetricType.MEAN}
  metrics = Metrics(config.report_interval, metric_types)
  replay = ReplayBuffer(config.replay_buffer_size)
  env = ZeldaEnvironment()

  device = torch.device('mps')
  agent = ZeldaAgent(config.hidden_size).to(device)
  optimizer = torch.optim.Adam(agent.parameters(), lr=config.lr)

  for episode_counter in range(1, config.num_episodes + 1):
    epsilon = max(0.00, 1. - 3 * float(episode_counter) / config.num_episodes)

    # Roll out an episode
    state, _ = env.reset()
    goal = state
    goal_pos = (120, 120, 7, 7)  # TODO: choose a real goal pos here
    for step in range(1, config.max_episode_length + 1):
      # Epsilon-greedy exploration strategy
      if np.random.uniform() > epsilon:
        with torch.no_grad():
          distances = agent(state[None], goal[None]).cpu()
          action = torch.argmin(distances).numpy()
      else:
        action = np.random.randint(0, NUM_ACTIONS)

      for _ in range(8):
        next_state, info = env.step([env.UP, env.DOWN, env.LEFT, env.RIGHT][action])
      done = env.pos_matches(info['pos'], goal_pos)
      replay.add_step(state, goal, action, next_state, done)
      state = next_state
      if done:
        break

    replay.commit()

    # Train the agent
    if len(replay) > config.batch_size:
      states, goals, actions, next_states, finished, mask = (x.squeeze(1) for x in replay.sample(config.batch_size))

      with torch.no_grad():
        best_future_distances = torch.clip(agent(next_states, goals).cpu().min(dim=-1).values * ~torch.as_tensor(finished), 0, config.max_episode_length)
      distances = agent(states, goals)[torch.arange(len(actions)), actions]
      loss = F.smooth_l1_loss(distances, best_future_distances.to(device) + 1)
      loss.backward()

      optimizer.step()
      optimizer.zero_grad()

    metrics.add({"Wins": done, "Episode Length": step, "Epsilon": epsilon})


# ENTRY POINT

if __name__ == '__main__':
  train(ZeldaConfig.from_cli())
