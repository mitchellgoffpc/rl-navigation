import numpy as np
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
from zelda.environment import ZeldaEnvironment

NUM_EPISODES = 100
NUM_STEPS = 500

def generate_episode(i):
    env = ZeldaEnvironment()

    frames = []
    actions = []
    map_positions = []
    screen_positions = []

    # Generate episode
    for _ in range(NUM_STEPS):
        action = np.random.choice([env.UP, env.DOWN, env.LEFT, env.RIGHT])
        obs, info = env.step(action)
        frames.append(obs)
        actions.append(action)
        map_positions.append(info['pos'][2:])
        screen_positions.append(info['pos'][:2])

    # Save episode
    dir_path = Path(__file__).parent / 'data'
    dir_path.mkdir(exist_ok=True)
    np.savez_compressed(dir_path / f'episode_{i}.npz', frames=frames, actions=actions, map_pos=map_positions, screen_pos=screen_positions)

if __name__ == "__main__":
    with Pool(8) as pool:
        list(tqdm(pool.imap(generate_episode, range(NUM_EPISODES)), total=NUM_EPISODES))
