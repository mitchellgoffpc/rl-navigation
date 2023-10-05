import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class ZeldaDataset(Dataset):
    def __init__(self, data_type='expert', max_distance=None):
        assert data_type in ['expert', 'random'], 'data type should be either expert or random'
        self.data_dir = Path(__file__).parent / data_type / 'data'
        self.episode_length = sum(1 for f in self.data_dir.iterdir() if f.name.startswith('episode_0_')) - 1
        self.num_episodes = sum(1 for x in self.data_dir.iterdir()) // (self.episode_length + 1)
        self.max_distance = max_distance or self.episode_length

    def __len__(self):
        return self.num_episodes * self.episode_length

    def get_step_idx(self, idx):
        episode_idx = idx // self.episode_length
        frame_idx = idx % self.episode_length
        return episode_idx, frame_idx

    def get_goal_idx(self, frame_idx):
        max_index = min(self.episode_length + 1, frame_idx + self.max_distance)
        return np.random.randint(frame_idx + 1, max_index)

    def load_steps(self, episode_idx, *step_idxs):
        return [np.load(self.data_dir / f'episode_{episode_idx}_step_{i}.npz') for i in step_idxs]


class ImitationDataset(ZeldaDataset):
    def __getitem__(self, idx):
        return torch.tensor([0])
        episode_idx, step_idx = self.get_step_idx(idx)
        goal_step_idx = self.get_goal_idx(step_idx)
        step, goal_step = self.load_steps(episode_idx, step_idx, goal_step_idx)

        state = torch.from_numpy(step['frame'])
        goal_state = torch.from_numpy(goal_step['frame'])
        action = torch.tensor(step['action'], dtype=torch.long)
        map_pos = torch.from_numpy(step['map_pos'])
        screen_pos = torch.from_numpy(step['screen_pos'])

        return state, goal_state, action, {'distance': goal_step_idx - step_idx, 'map_pos': map_pos, 'screen_pos': screen_pos}

class RLDataset(ZeldaDataset):
    def __getitem__(self, idx):
        episode_idx, step_idx = self.get_step_idx(idx)
        goal_step_idx = self.get_goal_idx(step_idx)
        step, goal_step, next_step = self.load_steps(episode_idx, step_idx, goal_step_idx, step_idx + 1)

        state = torch.from_numpy(step['frame'])
        goal_state = torch.from_numpy(goal_step['frame'])
        next_state = torch.from_numpy(next_step['frame'])
        action = torch.tensor(step['action'], dtype=torch.long)
        map_pos = torch.from_numpy(step['map_pos'])
        screen_pos = torch.from_numpy(step['screen_pos'])

        return state, goal_state, action, next_state, {'distance': goal_step_idx - step_idx, 'map_pos': map_pos, 'screen_pos': screen_pos}


if __name__ == '__main__':
    from tqdm import tqdm, trange
    from itertools import islice
    from torch.utils.data import DataLoader

    dataset = ImitationDataset('random')
    for i in trange(1000, desc="benchmarking dataset"):
        dataset[i]

    dataloader = DataLoader(dataset, batch_size=32, num_workers=8, shuffle=True)
    for _ in tqdm(dataloader, desc="benchmarking dataloader, n=8, bs=32"):
        pass
