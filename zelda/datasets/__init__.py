import re
import torch
import numpy as np
from pathlib import Path
from collections import Counter
from torch.utils.data import Dataset


class ZeldaDataset(Dataset):
  def __init__(self, data_type='expert', subsample=1, max_episode_len=None, max_goal_dist=None):
    assert data_type in ['expert', 'random'], 'data type should be either expert or random'
    self.data_dir = Path(__file__).parent / data_type / 'data'
    self.max_goal_dist = max_goal_dist or np.inf
    self.subsample = subsample

    episode_patterns = [re.search(r'episode_(\d+)_', fn.name) for fn in self.data_dir.iterdir()]
    episode_lens = Counter(int(x.group(1)) for x in episode_patterns if x)
    self.episode_lens = {k: v-1 for k,v in sorted(episode_lens.items())}
    if max_episode_len:
      self.episode_lens = {k: min(v, max_episode_len) for k,v in self.episode_lens.items()}
    self.cum_episode_lens = np.cumsum([0, *self.episode_lens.values()])

  def __len__(self):
    return int(sum(self.episode_lens.values()) * self.subsample)

  def get_step_idx(self, idx):
    episode_idx = np.searchsorted(self.cum_episode_lens, idx, side='right') - 1
    step_idx = idx - self.cum_episode_lens[episode_idx]
    return episode_idx, step_idx

  def get_goal_idx(self, episode_idx, step_idx):
    max_index = min(self.episode_lens[episode_idx], step_idx + self.max_goal_dist)
    return np.random.randint(step_idx + 1, max_index + 1)

  def get_step_data(self, episode_idx, step_idx):
    step = np.load(self.data_dir / f'episode_{episode_idx}_step_{step_idx}.npz')
    state = torch.from_numpy(step['frame'])
    action = torch.tensor(step['action'], dtype=torch.long)
    map_pos = torch.from_numpy(step['map_pos'])
    screen_pos = torch.from_numpy(step['screen_pos'])
    return state, action, map_pos, screen_pos


class ImitationDataset(ZeldaDataset):
  def __getitem__(self, idx):
    episode_idx, step_idx = self.get_step_idx(idx)
    goal_step_idx = self.get_goal_idx(episode_idx, step_idx)

    state, action, map_pos, screen_pos = self.get_step_data(episode_idx, step_idx)
    goal_state, *_ = self.get_step_data(episode_idx, goal_step_idx)
    return state, goal_state, action, {'distance': goal_step_idx - step_idx, 'map_pos': map_pos, 'screen_pos': screen_pos}

class RLDataset(ZeldaDataset):
  def __getitem__(self, idx):
    episode_idx, step_idx = self.get_step_idx(idx)
    goal_step_idx = self.get_goal_idx(episode_idx, step_idx)

    state, action, map_pos, screen_pos = self.get_step_data(episode_idx, step_idx)
    goal_state, *_ = self.get_step_data(episode_idx, goal_step_idx)
    next_state, *_ = self.get_step_data(episode_idx, step_idx + 1)
    return state, goal_state, action, next_state, {'distance': goal_step_idx - step_idx, 'map_pos': map_pos, 'screen_pos': screen_pos}

class ValidationDataset(ZeldaDataset):
  def __init__(self):
    super().__init__('expert')
    initial_map_pos = tuple(np.load(self.data_dir / f'episode_0_step_0.npz')['map_pos'])

    self.episode_lens = []
    for episode_idx in range(self.num_episodes):
      map_positions = [tuple(np.load(self.data_dir / f'episode_{episode_idx}_step_{i}.npz')['map_pos']) for i in range(self.episode_len)]
      self.episode_lens.append(next((i for i, map_pos in enumerate(map_positions) if map_pos != initial_map_pos), 0))
    self.cum_episode_lens = np.cumsum([0] + self.episode_lens)

  def __getitem__(self, idx):
    episode_idx, step_idx = self.get_step_idx(idx)
    goal_step_idx = self.get_goal_idx(episode_idx, step_idx)

    state, _, map_pos, screen_pos = self.get_step_data(episode_idx, step_idx)
    goal_state, _, goal_map_pos, _ = self.get_step_data(episode_idx, goal_step_idx)
    action = {(7, 6, 0): 0, (6, 7, 0): 2, (8, 7, 0): 3}[tuple(goal_map_pos.numpy())]
    return state, goal_state, action, {'distance': goal_step_idx - step_idx, 'map_pos': map_pos, 'screen_pos': screen_pos}
