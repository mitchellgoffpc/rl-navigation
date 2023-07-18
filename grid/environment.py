import numpy as np

class GridEnvironment:
  def __init__(self, grid_w, grid_h):
    self.grid_w, self.grid_h = grid_w, grid_h
    self.reset()

  def reset(self):
    self.grid = np.zeros((self.grid_h, self.grid_w), dtype=bool)
    self.goal = np.zeros((self.grid_h, self.grid_w), dtype=bool)
    self.pos = [np.random.randint(self.grid_h), np.random.randint(self.grid_w)]
    self.goal_pos = [np.random.randint(self.grid_h), np.random.randint(self.grid_w)]
    self.grid[tuple(self.pos)] = 1
    self.goal[tuple(self.goal_pos)] = 1
    return self.grid.copy(), self.goal.copy()

  def step(self, action):
    self.grid[tuple(self.pos)] = 0

    if action == 0:  # up
      if self.pos[0] > 0:
        self.pos[0] -= 1
    elif action == 1:  # down
      if self.pos[0] < self.grid_h - 1:
        self.pos[0] += 1
    elif action == 2:  # left
      if self.pos[1] > 0:
        self.pos[1] -= 1
    elif action == 3:  # right
      if self.pos[1] < self.grid_w - 1:
        self.pos[1] += 1

    self.grid[tuple(self.pos)] = 1
    return self.grid.copy(), 0, self.pos == self.goal_pos, {'pos': self.pos[:], 'goal_pos': self.goal_pos[:]}
