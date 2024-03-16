import random
import numpy as np
import gymnasium as gym
from collections import deque

def frontier(grid, x, y):
  f = set()
  height, width = grid.shape
  if x >= 0 and x < width and y >= 0 and y < height:
    if x > 1 and not grid[y, x-2]:
      f.add((x-2, y))
    if x + 2 < width and not grid[y, x+2]:
      f.add((x+2, y))
    if y > 1 and not grid[y-2, x]:
      f.add((x, y-2))
    if y + 2 < height and not grid[y+2, x]:
      f.add((x, y+2))

  return f

def neighbors(grid, x, y):
  n = set()
  height, width = grid.shape
  if x >= 0 and x < width and y >= 0 and y < height:
    if x > 1 and grid[y, x-2]:
      n.add((x-2, y))
    if x + 2 < width and grid[y, x+2]:
      n.add((x+2, y))
    if y > 1 and grid[y-2, x]:
      n.add((x, y-2))
    if y + 2 < height and grid[y+2, x]:
      n.add((x, y+2))

  return n

def generate_maze(width, height):
  s = set()
  grid = np.zeros((height, width), dtype=bool)
  x, y = (random.randint(0, width - 1), random.randint(0, height - 1))
  grid[y, x] = True
  fs = frontier(grid, x, y)
  for f in fs:
    s.add(f)
  while s:
    x, y = random.choice(tuple(s))
    s.remove((x, y))
    ns = neighbors(grid, x, y)
    if ns:
      nx, ny = random.choice(tuple(ns))
      grid[y, x] = True
      grid[(y + ny) // 2, (x + nx) // 2] = True

    fs = frontier(grid, x, y)
    for f in fs:
      s.add(f)

  return grid


class MazeEnv(gym.Env):
  MOVES = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}  # action: 0=up, 1=right, 2=down, 3=left

  def __init__(self, width, height):
    self.width = width
    self.height = height
    self.action_space = gym.spaces.Discrete(4)
    self.observation_space = gym.spaces.Box(low=0, high=3, shape=(height, width), dtype=np.uint8)
    self.reset()

  def is_valid_move(self, y, x):
    return 0 <= x < self.width and 0 <= y < self.height and self.maze[y, x]

  def get_random_positions(self):
    while True:
      yield random.randint(0, self.height - 1), random.randint(0, self.width - 1)

  def obs(self, position):
    obs = np.zeros((self.height, self.width), dtype=np.uint8)
    obs[:] = self.maze
    obs[position] = 2
    return obs

  def info(self):
    return {'position': self.current_position, 'goal': self.goal_position}

  def reset(self):
    self.maze = generate_maze(self.width, self.height)
    self.start_position = next(p for p in self.get_random_positions() if self.maze[p])
    self.goal_position = next(p for p in self.get_random_positions() if self.maze[p] and p != self.start_position)
    self.current_position = self.start_position
    return self.obs(self.current_position), self.obs(self.goal_position), self.info()

  def step(self, action):
    move = MazeEnv.MOVES[action]
    next_position = (self.current_position[0] + move[0], self.current_position[1] + move[1])
    if self.is_valid_move(*next_position):
      self.current_position = next_position
    if self.current_position == self.goal_position:
      return self.obs(self.current_position), 1, True, False, self.info()
    else:
      return self.obs(self.current_position), -1, False, False, self.info()

  def solve(self):
    queue = deque([(self.current_position, [])])
    visited = set([self.current_position])

    while queue:
      current, path = queue.popleft()
      if current == self.goal_position:
        return path

      for action, move in MazeEnv.MOVES.items():
        next_position = (current[0] + move[0], current[1] + move[1])
        if next_position not in visited and self.is_valid_move(*next_position):
          visited.add(next_position)
          queue.append((next_position, path + [action]))

    return []


if __name__ == "__main__":
  import sys
  import cv2
  import math
  import torch
  import pygame
  from pathlib import Path
  from maze.train_distance_policy import MLP
  assert len(sys.argv) == 3, "Usage: python environment.py <width> <height>"

  WHITE = (255, 255, 255)
  BLACK = (0, 0, 0)
  RED = (255, 0, 0)
  BLUE = (0, 0, 255)
  WIDTH, HEIGHT = int(sys.argv[1]), int(sys.argv[2])
  SCALE = 20
  PADDING = 80

  env = MazeEnv(WIDTH, HEIGHT)

  policy_model, distance_model = None, None
  if (Path(__file__).parent / 'checkpoints/policy-db.ckpt').exists():
    policy_model = MLP(math.prod(env.observation_space.shape), env.action_space.n).train()
    policy_model.load_state_dict(torch.load(Path(__file__).parent / 'checkpoints/policy-db.ckpt'))
  if (Path(__file__).parent / 'checkpoints/distance.ckpt').exists():
    distance_model = MLP(math.prod(env.observation_space.shape), env.action_space.n).train()  # Assuming the distance model predicts a single value
    distance_model.load_state_dict(torch.load(Path(__file__).parent / 'checkpoints/distance.ckpt'))

  pygame.init()
  screen = pygame.display.set_mode((WIDTH * SCALE, HEIGHT * SCALE + PADDING))
  clock = pygame.time.Clock()

  def draw(obs, goal, info, path=[]):
    screen.fill(BLACK)
    maze_img = np.zeros((*obs.shape, 3), dtype=np.uint8)
    maze_img[obs == 1] = (255, 255, 255)
    maze_img[obs == 2] = (255, 0, 0)
    maze_img[goal == 2] = (0, 0, 255)
    maze_img = cv2.resize(maze_img, (WIDTH * SCALE, HEIGHT * SCALE), interpolation=cv2.INTER_NEAREST)
    current_position = info['position']

    # Draw shortest path
    for step in path:
      action_position = (current_position[0] + MazeEnv.MOVES[step][0], current_position[1] + MazeEnv.MOVES[step][1])
      cv2.line(maze_img,
               (current_position[1] * SCALE + SCALE // 2, current_position[0] * SCALE + SCALE // 2),
               (action_position[1] * SCALE + SCALE // 2, action_position[0] * SCALE + SCALE // 2),
               (0, 255, 0), 2)
      current_position = action_position

    # Draw policy model predictions
    if policy_model is not None:
      maze_img = cv2.copyMakeBorder(maze_img, 0, PADDING, 0, 0, cv2.BORDER_CONSTANT, value=0)
      obs_tensor = torch.tensor(obs, dtype=torch.float32)[None]
      goal_tensor = torch.tensor(goal, dtype=torch.float32)[None]
      with torch.no_grad():
        action_probs = torch.softmax(policy_model(obs_tensor, goal_tensor), dim=1).squeeze().numpy()

      arrow_colors = [(255 * prob, 255 * prob, 255 * prob) for prob in action_probs]
      arrow_directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Up, Right, Down, Left
      arrow_base_size = SCALE // 3

      for direction, color in zip(arrow_directions, arrow_colors):
        dx, dy = direction
        center_x = WIDTH * SCALE // 4 + dx * arrow_base_size
        center_y = HEIGHT * SCALE + (PADDING // 2) + dy * arrow_base_size
        tip = (center_x + dx * arrow_base_size * 2, center_y + dy * arrow_base_size * 2)
        left_corner = (center_x + dy * arrow_base_size, center_y - dx * arrow_base_size)
        right_corner = (center_x - dy * arrow_base_size, center_y + dx * arrow_base_size)
        cv2.fillPoly(maze_img, [np.array([tip, left_corner, right_corner])], color)

    # Draw distance model predictions
    if distance_model is not None:
      obs_tensor = torch.as_tensor(obs)[None]
      goal_tensor = torch.as_tensor(goal)[None]
      with torch.no_grad():
        distance = distance_model(obs_tensor, goal_tensor).min().item()
      font = cv2.FONT_HERSHEY_SIMPLEX
      cv2.putText(maze_img, f"{distance:.2f}", (maze_img.shape[1] - 50, maze_img.shape[0] - 20), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    maze_surface = pygame.surfarray.make_surface(maze_img.swapaxes(0, 1))
    screen.blit(maze_surface, (0, 0))
    pygame.display.flip()


  state, goal, info = env.reset()
  draw(state, goal, info, path=env.solve())

  while True:
    action = None
    for event in pygame.event.get():
      if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)):
        pygame.quit()
        sys.exit()
      elif event.type == pygame.KEYDOWN:
        if event.key in (pygame.K_w, pygame.K_UP):
          action = 0
        elif event.key in (pygame.K_d, pygame.K_RIGHT):
          action = 1
        elif event.key in (pygame.K_s, pygame.K_DOWN):
          action = 2
        elif event.key in (pygame.K_a, pygame.K_LEFT):
          action = 3


    if action is None:
      continue

    obs, reward, done, _, info = env.step(action)
    draw(obs, goal, info, path=env.solve())

    if done:
      state, goal, info = env.reset()
      draw(state, goal, info, path=env.solve())
    clock.tick(5)
