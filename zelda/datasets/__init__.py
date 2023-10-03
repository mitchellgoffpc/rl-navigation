import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class ImitationDataset(Dataset):
    def __init__(self, data_type='expert', max_distance=None):
        assert data_type in ['expert', 'random'], 'data type should be either expert or random'
        data_dir = Path(__file__).parent / data_type / 'data'
        self.episode_paths = [f for f in data_dir.iterdir() if f.name.endswith('.npz')]

        first_episode = np.load(self.episode_paths[0])
        self.episode_length = len(first_episode['frames']) - 1
        self.max_distance = max_distance or self.episode_length

    def __len__(self):
        return len(self.episode_paths) * self.episode_length

    def __getitem__(self, idx):
        episode_idx = idx // self.episode_length
        frame_idx = idx % self.episode_length
        max_index = min(self.episode_length, frame_idx + self.max_distance)
        goal_state_idx = np.random.randint(frame_idx + 1, max_index)

        episode = dict(np.load(self.episode_paths[episode_idx]))
        state = torch.from_numpy(episode['frames'][frame_idx])
        goal_state = torch.from_numpy(episode['frames'][goal_state_idx])
        action = torch.tensor(episode['actions'][frame_idx], dtype=torch.long)
        map_pos = torch.from_numpy(episode['map_pos'][frame_idx])
        screen_pos = torch.from_numpy(episode['screen_pos'][frame_idx])

        return state, goal_state, action, {'distance': goal_state_idx - frame_idx, 'map_pos': map_pos, 'screen_pos': screen_pos}


class RLDataset(Dataset):
    def __init__(self, data_type='expert', max_distance=None):
        assert data_type in ['expert', 'random'], 'data type should be either expert or random'
        data_dir = Path(__file__).parent / data_type / 'data'
        self.episode_paths = [f for f in data_dir.iterdir() if f.name.endswith('.npz')]

        first_episode = np.load(self.episode_paths[0])
        self.episode_length = len(first_episode['frames']) - 1
        self.max_distance = max_distance or self.episode_length

    def __len__(self):
        return len(self.episode_paths) * self.episode_length

    def __getitem__(self, idx):
        episode_idx = idx // self.episode_length
        frame_idx = idx % self.episode_length
        max_index = min(self.episode_length, frame_idx + self.max_distance)
        goal_state_idx = np.random.randint(frame_idx + 1, max_index)

        episode = dict(np.load(self.episode_paths[episode_idx]))
        state = torch.from_numpy(episode['frames'][frame_idx])
        goal_state = torch.from_numpy(episode['frames'][goal_state_idx])
        next_state = torch.from_numpy(episode['frames'][frame_idx + 1])
        action = torch.tensor(episode['actions'][frame_idx], dtype=torch.long)
        map_pos = torch.from_numpy(episode['map_pos'][frame_idx])
        screen_pos = torch.from_numpy(episode['screen_pos'][frame_idx])

        return state, goal_state, action, next_state, {'distance': goal_state_idx - frame_idx, 'map_pos': map_pos, 'screen_pos': screen_pos}


if __name__ == '__main__':
    from tqdm import trange
    dataset = ImitationDataset('random')
    for i in trange(100):
        dataset[i]
