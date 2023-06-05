import numpy as np
from typing import List, Tuple

class ReplayBuffer:
  current_size: int
  max_size: int
  episode: List[Tuple[np.ndarray, ...]]
  buffer: List[Tuple[np.ndarray, ...]]

  def __init__(self, size):
    self.current_size = 0
    self.max_size = size
    self.buffer = []
    self.episode = []

  def add_step(self, *data:np.ndarray):
    self.episode.append(data)

  def add_episode(self, *data:np.ndarray):
    if len(self.buffer) < self.max_size:
      self.buffer.append(data)
    else:
      self.buffer[np.random.randint(0, len(self.buffer) - 1)] = data

  def commit(self):
    assert len(self.episode) > 0, "Can't commit an empty episode!"
    columns = []
    for i in range(len(self.episode[0])):
      columns.append(np.stack([step[i] for step in self.episode], axis=0))
    self.add_episode(*columns)
    self.episode = []

  def sample(self, bs:int, n_steps:int = 1) -> Tuple[np.ndarray, ...]:
    assert len(self.buffer) > 0, "Can't sample from an empty replay buffer!"
    batch_idxs = np.random.randint(0, len(self.buffer), size=bs)
    ep_lengths = np.array([len(self.buffer[i][0]) for i in batch_idxs])
    start_idxs = np.random.randint(0, ep_lengths, size=bs)
    end_idxs = np.minimum(start_idxs + n_steps, ep_lengths)

    columns = []
    for c in range(len(self.buffer[0])):
      shape = self.buffer[0][c].shape[1:]
      dtype = self.buffer[0][c].dtype
      column = np.zeros((bs, n_steps, *shape), dtype=dtype)
      for i,(b,s,e) in enumerate(zip(batch_idxs, start_idxs, end_idxs)):
        column[i,:e-s] = self.buffer[b][c][s:e]
      columns.append(column)

    mask = np.zeros((bs, n_steps), dtype=bool)
    for i,(b,s,e) in enumerate(zip(batch_idxs, start_idxs, end_idxs)):
      mask[i,:e-s] = 1
    columns.append(mask)

    return columns


# TESTING

if __name__ == '__main__':
  import time
  replay = ReplayBuffer(1024)

  for _ in range(1024+1):
    for step in range(10):
      replay.add_step(np.zeros((256, 256)), np.ones(6, dtype=np.uint8), np.array(2.))
    replay.commit()

  states, actions, rewards, mask = replay.sample(8, 5)
  assert states.shape == (8, 5, 256, 256)
  assert np.all(states == 0)
  assert actions.shape == (8, 5, 6)
  assert actions.dtype == np.uint8
  assert np.all(actions[mask] == 1)
  assert rewards.shape == (8, 5)
  assert np.all(rewards[mask] == 2.)
  assert mask.shape == (8, 5)
  assert mask.dtype == bool
  print("All tests passed")

  print("Benchmarking...")
  st = time.perf_counter()
  for _ in range(100):
    replay.sample(64, 8)
  et = time.perf_counter()
  print(f"Benchmark: {(et-st)/100*1000:.2f}ms/sample")
