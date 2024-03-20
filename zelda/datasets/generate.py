import numpy as np
import multiprocessing
from tqdm import tqdm
from pathlib import Path
from zelda.environment import ZeldaEnvironment

NUM_EPISODES = 100
NUM_STEPS = 500
NUM_REPEATS = 8

def generate_random_episode(i):
  env = ZeldaEnvironment()
  obs, info = env.reset()
  dir_path = Path(__file__).parent / 'random'
  dir_path.mkdir(exist_ok=True)

  for j in range(NUM_STEPS):
    action = env.action_space.sample()
    np.savez_compressed(dir_path / f'episode_{i}_step_{j}.npz', frame=obs, action=action, pos=info['pos'])
    for _ in range(NUM_REPEATS):
      obs, _, _, _, info = env.step(action)

def generate_random_episodes():
    with multiprocessing.Pool(8) as pool:
        list(tqdm(pool.imap(generate_random_episode, range(NUM_EPISODES)), total=NUM_EPISODES))

def generate_expert_episodes():
  import pygame

  KEYS_TO_ACTIONS = {
    pygame.K_w: ZeldaEnvironment.UP,
    pygame.K_s: ZeldaEnvironment.DOWN,
    pygame.K_a: ZeldaEnvironment.LEFT,
    pygame.K_d: ZeldaEnvironment.RIGHT}

  pygame.init()
  screen = pygame.display.set_mode((256, 240))
  clock = pygame.time.Clock()
  env = ZeldaEnvironment()

  data_dir = Path(__file__).parent / 'expert'
  data_dir.mkdir(exist_ok=True)
  episodes = [f.name for f in data_dir.iterdir() if f.name.endswith('.npz')]
  indices = [int(ep.split('_')[1].split('.')[0]) for ep in episodes]
  start_idx = max(indices) if indices else -1

  for i in range(start_idx + 1, NUM_EPISODES):
    step = 0
    obs, info = env.reset()
    while step < NUM_STEPS:
      clock.tick(60)  # Delay to achieve 60 fps

      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          pygame.quit()
          sys.exit()

      keys = pygame.key.get_pressed()
      for action, key in enumerate(KEYS_TO_ACTIONS.keys()):
        if keys[key]:
          obs, info = env.step(KEYS_TO_ACTIONS[key])
          fn = data_dir / f'episode_{i}_step_{step}.npz'
          np.savez_compressed(fn, frame=obs, action=action, pos=info['pos'])
          step += 1

      screen.blit(pygame.surfarray.make_surface(obs.swapaxes(0, 1)), (0, 0))
      pygame.display.flip()
