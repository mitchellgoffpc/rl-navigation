import random
import numpy as np
import torch
from tqdm import trange

def batch(data, indices=None, device=torch.device('cpu')):
  if indices is not None:
    data = [data[i] for i in indices]
  return tuple(torch.as_tensor(np.array(x), device=device) for x in zip(*data))

def generate_data(env, num_episodes, num_steps, split="train"):
  steps = []
  for _ in trange(num_episodes, desc=f"Generating {split} set"):
    state, goal, _ = env.reset()
    for i in range(num_steps):
      # correct_action = env.solve()[0]
      solution = env.solve()
      correct_action = solution[0] if solution else random.randrange(env.action_space.n)
      action = random.randrange(env.action_space.n)
      next_state, _, done, _, _ = env.step(action)
      steps.append((state, goal, action, next_state, done, correct_action))
      state = next_state
      # if done:
      #   break

  return steps

def evaluate_policy(env, policy, num_episodes, num_steps, device=torch.device('cpu')):
  sucesses = 0
  for _ in trange(num_episodes, desc="Evaluating policy", leave=False):
    state, goal, _ = env.reset()
    for _ in range(num_steps):
      action = policy(torch.as_tensor(state)[None].to(device), torch.as_tensor(goal)[None].to(device))
      state, _, done, _, _ = env.step(action)
      if done:
        sucesses += 1
        break

  return sucesses / num_episodes
