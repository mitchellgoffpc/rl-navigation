import numpy as np
from pathlib import Path
from gym.spaces import Box
from common.environments.nes import NESEnvironment

class ZeldaEnvironment(NESEnvironment):
  def __init__(self):
    super().__init__(Path(__file__).parent / f"zelda.nes")
    super().reset()
    self.wait(40)
    self.step(self.START, wait=20)
    self.step(self.START, wait=10)
    self.step(self.A, wait=10)
    for _ in range(3):
      self.step(self.SELECT, wait=1)
    self.step(self.START, wait=20)
    self.step(self.START, wait=120)
    self._backup()

  @property
  def action_space(self):
    return Discrete(4)

  @property
  def observation_space(self):
    return Box(low=0, high=255, shape=(240, 256, 3), dtype=np.uint8)

  def reset(self):
    self._restore()
    return self.step(0)

  def step(self, *args, **kwargs):
    obs = super().step(*args, **kwargs)
    info = {'screen_pos': self.screen_pos, 'map_pos': self.map_pos, 'pos': (*self.screen_pos, *self.map_pos)}
    return obs, info

  def pos_matches(self, a, b):
    ax, ay, amx, amy, aml = map(int, a)
    bx, by, bmx, bmy, bml = map(int, b)
    return abs(ax - bx) <= 4 and abs(ay - by) <= 4 and (amx, amy, aml) == (bmx, bmy, bml)

  @property
  def screen_pos(self):
    return int(self.ram[0x70]), int(self.ram[0x84])  # uint8 break subtraction

  @property
  def map_pos(self):
    # NOTE: 0x0609 is the song type, this is the only way I can find to reliably determine which map level you're on.
    return int(self.ram[0xEB] % 0x10), int(self.ram[0xEB] // 0x10), int(self.ram[0x0609] != 1)
