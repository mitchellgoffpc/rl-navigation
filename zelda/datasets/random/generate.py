import numpy as np
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
from zelda.environment import ZeldaEnvironment

NUM_EPISODES = 100
NUM_STEPS = 500
NUM_REPEATS = 8

def generate_episode(i):
    env = ZeldaEnvironment()
    obs, info = env.reset()
    dir_path = Path(__file__).parent / 'data'
    dir_path.mkdir(exist_ok=True)
    actions = [env.UP, env.DOWN, env.LEFT, env.RIGHT]

    for j in range(NUM_STEPS):
        action = np.random.randint(len(actions))
        np.savez_compressed(dir_path / f'episode_{i}_step_{j}.npz', frame=obs, action=action, map_pos=info['map_pos'], screen_pos=info['screen_pos'])
        for _ in range(NUM_REPEATS):
            obs, info = env.step(actions[action])

if __name__ == "__main__":
    with Pool(8) as pool:
        list(tqdm(pool.imap(generate_episode, range(NUM_EPISODES)), total=NUM_EPISODES))
