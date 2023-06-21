from nes_py import NESEnv as RawNESEnv

class NESEnvironment:
  A      = 1 << 0
  B      = 1 << 1
  SELECT = 1 << 2
  START  = 1 << 3
  UP     = 1 << 4
  DOWN   = 1 << 5
  LEFT   = 1 << 6
  RIGHT  = 1 << 7

  def __init__(self, path):
    self.nes = RawNESEnv(str(path))

  def reset(self):
    return self.nes.reset()

  def wait(self, n_frames):
    for _ in range(n_frames):
      frame, _, _, _ = self.nes.step(0)
    return frame.copy()

  def step(self, action, wait=None):
    if wait is not None:
      self.nes.step(action)
      return self.wait(wait)
    else:
      # ugh who thought it was a good idea to reuse the obs buffer...
      obs, state, data, info = self.nes.step(action)
      return obs.copy(), state, data, info
