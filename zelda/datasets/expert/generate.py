import sys
import pygame
import numpy as np
from pathlib import Path
from zelda.environment import ZeldaEnvironment

NUM_EPISODES  = 20
NUM_STEPS = 500
KEYS_TO_ACTIONS = {
    pygame.K_w: ZeldaEnvironment.UP,
    pygame.K_a: ZeldaEnvironment.LEFT,
    pygame.K_s: ZeldaEnvironment.DOWN,
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
        frames = []
        actions = []
        map_pos = []
        screen_pos = []
        step = 0

        obs, info = env.reset()
        while step < NUM_STEPS:
            clock.tick(60)  # Delay to achieve 60 fps

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            keys = pygame.key.get_pressed()
            for key, action in KEYS_TO_ACTIONS.items():
                if keys[key]:
                    obs, info = env.step(action)
                    frames.append(obs)
                    actions.append(action)
                    map_pos.append(info['pos'][:2])
                    screen_pos.append(info['pos'][2:])
                    step += 1

            screen.blit(pygame.surfarray.make_surface(obs.swapaxes(0, 1)), (0, 0))
            pygame.display.flip()

        np.savez_compressed(data_dir / f'episode_{i}.npz', frames=frames, actions=actions, map_pos=map_pos, screen_pos=screen_pos)