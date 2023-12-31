import gym
import numpy as np
from pathlib import Path
from common.environments.nes import NESEnvironment

class ZeldaEnvironment(NESEnvironment):
  ACTIONS = [NESEnvironment.UP, NESEnvironment.DOWN, NESEnvironment.LEFT, NESEnvironment.RIGHT]

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
  def observation_space(self):
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


# REGISTER GYM ENVIRONMENT
gym.register('zelda', ZeldaEnvironment)
