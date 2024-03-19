import numpy as np
import gymnasium as gym
from pathlib import Path
from common.environments.nes import NESEnvironment

class ZeldaEnvironment(NESEnvironment):
  ACTIONS = {None: 0, 0: NESEnvironment.UP, 1: NESEnvironment.DOWN, 2: NESEnvironment.LEFT, 3: NESEnvironment.RIGHT}

  def __init__(self):
    super().__init__(Path(__file__).parent / f"zelda.nes")
    super().reset()
    self.wait(40)
    self.act(self.START, wait=20)
    self.act(self.START, wait=10)
    self.act(self.A, wait=10)
    for _ in range(3):
      self.act(self.SELECT, wait=1)
    self.act(self.START, wait=20)
    self.act(self.START, wait=120)
    self._backup()

  @property
  def action_space(self):
    return gym.spaces.Discrete(4)

  @property
  def obs_space(self):
    return gym.spaces.Box(low=0, high=255, shape=(240, 256, 3), dtype=np.uint8)

  def reset(self, seed=None, options=None):
    self._restore()
    return super().step(0), self.get_info()

  def step(self, action, *args, **kwargs):
    obs = super().step(self.ACTIONS[action], *args, **kwargs)
    return obs, 0, False, False, self.get_info()

  def act(self, action, *args, **kwargs):
    obs = super().step(action, *args, **kwargs)
    return obs, self.get_info()

  def get_info(self):
    return {'screen_pos': self.screen_pos, 'map_pos': self.map_pos, 'pos': (*self.screen_pos, *self.map_pos)}

  @property
  def screen_pos(self):
    return int(self.ram[0x70]), int(self.ram[0x84])  # uint8 break subtraction

  @property
  def map_pos(self):
    # NOTE: 0x0609 is the song type, this is the only way I can find to reliably determine which map level you're on.
    return int(self.ram[0xEB] % 0x10), int(self.ram[0xEB] // 0x10), int(self.ram[0x0609] != 1)

  @classmethod
  def pos_matches(cls, a, b, tolerance=4):
    ax, ay, amx, amy, aml = map(int, a)
    bx, by, bmx, bmy, bml = map(int, b)
    return abs(ax - bx) <= tolerance and abs(ay - by) <= tolerance and (amx, amy, aml) == (bmx, bmy, bml)


gym.register('zelda', ZeldaEnvironment)


# ENTRY POINT
if __name__ == '__main__':
  import pygame
  import sys
  import cv2

  env = ZeldaEnvironment()

  pygame.init()
  screen = pygame.display.set_mode((512 * 2 + 10, 480))
  pygame.display.set_caption('Zelda Environment')
  clock = pygame.time.Clock()

  def reset(env):
    env.reset()
    for _ in range(20):
      action = env.action_space.sample()
      for _ in range(8):
        goal, _, _, _, goal_info = env.step(action)
    obs, info = env.reset()
    return obs, goal, info, goal_info

  def draw(frame, pos):
    frame = cv2.resize(frame, (512, 480), interpolation=cv2.INTER_NEAREST)
    surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    screen.blit(surface, pos)

  # Game loop
  obs, goal, info, goal_info = reset(env)
  while True:
    action = None
    if any(event.type == pygame.QUIT for event in pygame.event.get()):
      break

    keys = pygame.key.get_pressed()
    if keys[pygame.K_w] or keys[pygame.K_UP]:
      action = 0
    elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
      action = 1
    elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
      action = 2
    elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
      action = 3
    elif keys[pygame.K_r]:
      obs, goal, info = reset(env)
      continue
    elif keys[pygame.K_ESCAPE]:
      break

    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
      obs, goal, info = reset(env)

    # Draw the screen
    if env.pos_matches(info['pos'], goal_info['pos']):
        screen.fill((0, 255, 0))
    else:
        screen.fill((0, 0, 0))

    draw(obs, (0, 0))
    draw(goal, (512 + 10, 0))
    pygame.display.flip()
    clock.tick(60)

  pygame.quit()
  sys.exit()
