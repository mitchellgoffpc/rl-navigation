from nes_py import NESEnv

class NESEnvironment(NESEnv):
  A      = 1 << 0
  B      = 1 << 1
  SELECT = 1 << 2
  START  = 1 << 3
  UP     = 1 << 4
  DOWN   = 1 << 5
  LEFT   = 1 << 6
  RIGHT  = 1 << 7

  def __init__(self, path):
    super().__init__(str(path))

  def reset(self):
    return super().reset()

  def wait(self, n_frames):
    for _ in range(n_frames):
      frame, _, _, _ = super().step(0)
    return frame.copy()

  def step(self, action, wait=None):
    if wait is not None:
      super().step(action)
      return self.wait(wait)
    else:
      # ugh who thought it was a good idea to reuse the obs buffer...
      obs, reward, done, info = super().step(action)
      return obs.copy()
