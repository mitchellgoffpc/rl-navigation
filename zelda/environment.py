from pathlib import Path
from common.environments.nes import NESEnvironment

class ZeldaEnvironment(NESEnvironment):
  def __init__(self):
    super().__init__(Path(__file__).parent / f"zelda.nes")

  def reset(self):
    super().reset()
    self.wait(40)
    self.step(self.START, wait=20)
    self.step(self.START, wait=10)
    self.step(self.A, wait=10)
    for _ in range(3):
      self.step(self.SELECT, wait=1)
    self.step(self.START, wait=20)
    return self.step(self.START, wait=120)

  def step(self, *args, **kwargs):
    obs = super().step(*args, **kwargs)
    pos_x, pos_y = self.screen_pos
    map_x, map_y = self.map_pos
    info = {'pos': (pos_x, pos_y, map_x, map_y)}
    return obs, info

  def pos_matches(self, a, b):
    ax, ay, amx, amy = a
    bx, by, bmx, bmy = b
    return abs(ax - bx) <= 8 and abs(ay - by) <= 8 and (amx, amy) == (bmx, bmy)

  @property
  def memory(self):
    return self.ram

  @property
  def screen_pos(self):
    return self.memory[0x70], self.memory[0x84]

  @property
  def map_pos(self):
    return self.memory[0xEB] % 0x10, self.memory[0xEB] // 0x10
