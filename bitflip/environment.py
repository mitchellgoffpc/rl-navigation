import numpy as np

class BitflipEnvironment:
  bit_length: int
  state: np.ndarray
  goal: np.ndarray

  def __init__(self, bit_length:int):
    self.bit_length = bit_length
    self.reset()

  def reset(self):
    self.state = self.random_state()
    self.goal = self.random_state()
    return self.state, self.goal

  def random_state(self):
    state = np.random.randint(0, 2 ** self.bit_length)
    return np.array([(state >> i) & 1 for i in range(self.bit_length)], dtype=bool)

  def step(self, action):
    self.state[action] = ~self.state[action]
    return self.state.copy(), np.all(self.state == self.goal)
