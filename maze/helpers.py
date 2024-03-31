import random
import torch
from tqdm import trange

def generate_data(env, num_episodes, num_steps, split="train"):
  steps = []
  for _ in trange(num_episodes, desc=f"Generating {split} set"):
    episode = []
    goal_reached = False
    state, goal, _ = env.reset()
    for _ in range(num_steps):
      correct_action = env.solve()[0]
      action = random.randrange(env.action_space.n)
      next_state, _, done, _, _ = env.step(action)
      episode.append((state, goal, action, next_state, done, correct_action))
      if done:
        goal_reached = True
        break
      state = next_state
    steps.extend([(*step, len(episode) - i - 1, goal_reached) for i, step in enumerate(episode)])

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
