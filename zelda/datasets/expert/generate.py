import sys
import pygame
import numpy as np
from pathlib import Path
from zelda.environment import ZeldaEnvironment

NUM_EPISODES  = 20
NUM_STEPS = 500
KEYS_TO_ACTIONS = {
    pygame.K_w: ZeldaEnvironment.UP,
    pygame.K_s: ZeldaEnvironment.DOWN,
    pygame.K_a: ZeldaEnvironment.LEFT,
    pygame.K_d: ZeldaEnvironment.RIGHT}


if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((256, 240))
    clock = pygame.time.Clock()
    env = ZeldaEnvironment()

    data_dir = Path(__file__).parent / 'data'
    data_dir.mkdir(exist_ok=True)
    episodes = [f.name for f in data_dir.iterdir() if f.name.endswith('.npz')]
    indices = [int(ep.split('_')[1].split('.')[0]) for ep in episodes]
    start_idx = max(indices) if indices else -1

    # Generate episodes
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
                    np.savez_compressed(fn, frame=obs, action=action, map_pos=info['map_pos'], screen_pos=info['screen_pos'])
                    step += 1

            screen.blit(pygame.surfarray.make_surface(obs.swapaxes(0, 1)), (0, 0))
            pygame.display.flip()
