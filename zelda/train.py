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
  batch_size = 512
  num_episodes = 20000
  max_episode_length = 10
  replay_buffer_size = 1000
  report_interval = 10


def train(config):
  metric_types = {"Wins": MetricType.SUM, "Episode Length": MetricType.MEAN, "Epsilon": MetricType.MEAN}
  metrics = Metrics(config.report_interval, metric_types)
  replay = ReplayBuffer(config.replay_buffer_size)
  env = ZeldaEnvironment()
  agent = ZeldaAgent(config.hidden_size)
  optimizer = torch.optim.Adam(agent.parameters(), lr=config.lr)

  for episode_counter in range(1, config.num_episodes + 1):
    epsilon = max(0.00, 1. - 3 * float(episode_counter) / config.num_episodes)

    # Roll out an episode
    state = env.reset()
    goal = state
    for step in range(1, config.max_episode_length + 1):
      with torch.no_grad():
        distances = agent(state[None], goal[None])

      # Epsilon-greedy exploration strategy
      if np.random.uniform() > epsilon:
        action = torch.argmin(distances).numpy()
      else:
        action = np.random.randint(0, NUM_ACTIONS)

      next_state, _, _, _ = env.step(action)
      replay.add_step(state, goal, action, next_state)
      state = next_state

    replay.commit()

    # # Train the agent
    # if replay.num_steps() > config.batch_size:
    #   states, goals, actions, next_states, finished, _ = (x.squeeze() for x in replay.sample(config.batch_size))
    #
    #   with torch.no_grad():
    #     best_future_distances = torch.clip(agent(next_states, goals).min(dim=-1).values * ~finished, 0, config.bit_length)
    #   distances = agent(states, goals)[torch.arange(len(actions)), actions]
    #   loss = F.smooth_l1_loss(distances, best_future_distances + 1)
    #   loss.backward()
    #
    #   optimizer.step()
    #   optimizer.zero_grad()
    #
    # metrics.add({"Wins": done, "Episode Length": step, "Epsilon": epsilon})
    metrics.add({"Episode Length": step})


# ENTRY POINT

if __name__ == '__main__':
  train(ZeldaConfig.from_cli())
