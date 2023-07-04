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

  @property
  def memory(self):
    return self.nes.ram

  @property
  def screen_pos(self):
    return self.memory[0x70], self.memory[0x84]

  @property
  def map_pos(self):
    return self.memory[0xEB] % 0x10, self.memory[0xEB] // 0x10
