import time
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass

from common.metrics import Metrics, MetricType
from common.replay import ReplayBuffer
from common.config import BaseTrainingConfig
from zelda.models import ZeldaAgent, NUM_ACTIONS, INPUT_SIZE
from zelda.environment import ZeldaEnvironment

@dataclass
class ZeldaConfig(BaseTrainingConfig):
  lr: float = 0.0003
  batch_size: int = 32
  num_episodes: int = 1000
  max_episode_length: int = 6
  replay_buffer_size: int = 1000
  goal_replacement_prob: float = 0.8
  num_train_steps: int = 1
  report_interval: int = 10
  device_name: str = 'mps'


def train(config):
  metric_types = {"Wins": MetricType.SUM, "Episode Length": MetricType.MEAN, "Epsilon": MetricType.MEAN}
  metrics = Metrics(config.report_interval, metric_types)
  replay = ReplayBuffer(config.replay_buffer_size)
  env = ZeldaEnvironment()
  agent = ZeldaAgent().to(config.device)
  optimizer = torch.optim.Adam(agent.parameters(), lr=config.lr)

  # Come up with a goal
  env.reset()
  for _ in range(2*8):
    goal, info = env.step(env.UP)
  for _ in range(1*8):
    goal, info = env.step(env.LEFT)
  goal_pos = info['pos']
  # import matplotlib.pyplot as plt
  # plt.imshow(goal)
  # plt.show()

  for episode_counter in range(1, config.num_episodes + 1):
    epsilon = max(0.00, 1. - 2 * float(episode_counter) / config.num_episodes)

    # Roll out an episode
    state, _ = env.reset()
    if len(replay) and np.random.uniform() < 0.5:
      _, _, _, goal, _, goal_pos, _ = (x.squeeze() for x in replay.sample(1))
    # else:
    #   goal, goal_pos = np.zeros(INPUT_SIZE, dtype=np.uint8), (120, 120, 7, 7)

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
      replay.add_step(state, goal, action, next_state, done, info['pos'])
      state = next_state
      if done:
        break

    replay.commit()

    # Train the agent
    if len(replay) > config.batch_size:
      for _ in range(config.num_train_steps):
        # states, goals, actions, next_states, finished, _, mask = (x.squeeze(1) for x in replay.sample(config.batch_size))
        states, goals, actions, next_states, finished, _, mask = replay.sample(config.batch_size, config.max_episode_length)
        use_alt_goals = np.random.uniform(size=config.batch_size) < config.goal_replacement_prob
        alt_goal_idxs = np.random.randint(0, np.sum(mask, axis=-1))
        goals = np.where(use_alt_goals[:,None,None,None], states[np.arange(len(states)), alt_goal_idxs], goals[:,0])
        finished = np.where(use_alt_goals, alt_goal_idxs <= 1, finished[:,0])
        states, actions, next_states = (x[:,0] for x in (states, actions, next_states))

        with torch.no_grad():
          best_future_distances = torch.clip(agent(next_states, goals).cpu().min(dim=-1).values * ~torch.as_tensor(finished), 0, config.max_episode_length)
        distances = agent(states, goals)[torch.arange(len(actions)), actions]
        loss = F.smooth_l1_loss(distances, best_future_distances.to(config.device) + 1)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    metrics.add({"Wins": done, "Episode Length": step, "Epsilon": epsilon})


# ENTRY POINT

if __name__ == '__main__':
  train(ZeldaConfig.from_cli())
