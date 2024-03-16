import random
import numpy as np
from tqdm import trange

def batch(data, indices=None):
  if indices is not None:
    data = [data[i] for i in indices]
  return tuple(map(np.array, zip(*data)))

def generate_data(env, num_episodes, num_steps, split="train"):
  steps = []
  for _ in trange(num_episodes, desc=f"Generating {split} set"):
    state, goal, _ = env.reset()
    for i in range(num_steps):
      # correct_action = env.solve()[0]
      solution = env.solve()
      correct_action = solution[0] if solution else random.randrange(env.action_space.n)
      action = random.randrange(env.action_space.n)
      next_state, reward, done, _, _ = env.step(action)
      new_solution = env.solve()
      steps.append((state, goal, action, next_state, correct_action, len(solution), len(new_solution)))
      state = next_state
      # if done:
      #   break

  return steps

def evaluate_policy(env, policy, num_episodes, num_steps):
  sucesses = 0
  for _ in trange(num_episodes, desc="Evaluating policy", leave=False):
    state, goal, _ = env.reset()
    for _ in range(num_steps):
      action = policy(state, goal)
      state, _, done, _, _ = env.step(action)
      if done:
        sucesses += 1
        break

  return sucesses / num_episodes
