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
        goal_step_idx = self.get_goal_idx(step_idx)

        state, action, map_pos, screen_pos = self.get_step_data(episode_idx, step_idx)
        goal_state, *_ = self.get_step_data(episode_idx, goal_step_idx)
        return state, goal_state, action, {'distance': goal_step_idx - step_idx, 'map_pos': map_pos, 'screen_pos': screen_pos}

class RLDataset(ZeldaDataset):
    def __getitem__(self, idx):
        episode_idx, step_idx = self.get_step_idx(idx)
        goal_step_idx = self.get_goal_idx(step_idx)

        state, action, map_pos, screen_pos = self.get_step_data(episode_idx, step_idx)
        goal_state, *_ = self.get_step_data(episode_idx, goal_step_idx)
        next_state, *_ = self.get_step_data(episode_idx, step_idx + 1)
        return state, goal_state, action, next_state, {'distance': goal_step_idx - step_idx, 'map_pos': map_pos, 'screen_pos': screen_pos}

class ValidationDataset(ZeldaDataset):
    def __init__(self):
        super().__init__('expert')
        initial_map_pos = np.load(self.data_dir / f'episode_0_step_0.npz')['map_pos']

        self.episode_lengths = []
        for episode_idx in range(self.num_episodes):
            steps = (np.load(self.data_dir / f'episode_{episode_idx}_step_{i}.npz') for i in range(self.episode_length))
            self.episode_lengths.append(next(i for i, step in enumerate(steps) if step['map_pos'] != initial_map_pos))
        self.cum_episode_lengths = np.cumsum([0] + self.episode_lengths)

    def __len__(self):
        return sum(self.episode_lengths)

    def __getitem__(self, idx):
        episode_idx, frame_idx = self.get_step_idx(idx)
        goal_step_idx = self.get_goal_idx(episode_idx, step_idx)

        state, action, map_pos, screen_pos = self.get_step_data(episode_idx, step_idx)
        goal_state, *_ = self.get_step_data(episode_idx, goal_step_idx)
        return state, goal_state, action, {'distance': goal_step_idx - step_idx, 'map_pos': map_pos, 'screen_pos': screen_pos}

    def get_step_idx(self, idx):
        episode_idx = np.searchsorted(self.cum_episode_lengths, idx, side='right') - 1
        frame_idx = idx - self.cum_episode_lengths[episode_idx]
        return episode_idx, frame_idx

    def get_goal_idx(self, episode_idx, frame_idx):
        return np.random.randint(frame_idx + 1, self.episode_lengths[episode_idx])


if __name__ == '__main__':
    import random
    import argparse
    import matplotlib.pyplot as plt
    from tqdm import tqdm, trange
    from itertools import islice
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument('command', nargs='?', default='visualize', choices=['visualize', 'benchmark'])
    args = parser.parse_args()

    # Visualize data
    if args.command == 'visualize':
        dataset = ImitationDataset('random')
        for i in range(5):
            image, goal_state, action, info = random.choice(dataset)
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(image.numpy())
            ax[0].set_title(f'State\nAction: {action}')
            ax[1].imshow(goal_state.numpy())
            ax[1].set_title(f'Goal State\nDistance: {info["distance"]}, map_pos: {info["map_pos"].tolist()}, screen_pos: {info["screen_pos"].tolist()}')
            plt.tight_layout()
            plt.show()

    # Benchmark dataset
    elif args.command == 'benchmark':
        dataset = ImitationDataset('random')
        for i in trange(1000, desc="benchmarking dataset"):
            dataset[i]

        # Benchmark dataloader
        dataloader = DataLoader(dataset, batch_size=32, num_workers=8, shuffle=True)
        for _ in tqdm(dataloader, desc="benchmarking dataloader, n=8, bs=32"):
            pass
