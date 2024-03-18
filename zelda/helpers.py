import random
import torch
import numpy as np
from tqdm import trange


def generate_rollout(env, num_steps, num_repeats=8):
  rollout = []
  obs, info = env.reset()
  for _ in range(num_steps):
    action = env.action_space.sample()
    rollout.append((obs, action, info))
    for _ in range(num_repeats):
      obs, _, _, _, info = env.step(action)
  return rollout

def generate_goals(env, num_goals, num_steps):
  goal_states, goal_positions = [], []
  for _ in range(num_goals):
    rollout = generate_rollout(env, num_steps)
    obs, _, info = random.choice(rollout)
    goal_states.append(obs)
    goal_positions.append(info['pos'])
  return goal_states, goal_positions

def evaluate_model(env, policy, goals, goal_positions, num_steps, num_repeats=8):
  BS = 1
  finished = np.zeros(len(goals), dtype=bool)
  for offset in trange(0, len(goals), BS, desc="Evaluating model"):
    obs, _ = env.reset()
    batch_goals = np.stack(goals[offset:offset+BS], axis=0)
    batch_goal_pos = goal_positions[offset:offset+BS]
    batch_finished = finished[offset:offset+BS]

    for _ in range(num_steps):
      with torch.no_grad():
        action = policy(obs[None], batch_goals)
      for _ in range(num_repeats):
        obs, _, _, _, info = env.step(action)
      at_goal = [env.pos_matches(pos, goal_pos, tolerance=10) for pos, goal_pos in zip([info['pos']], batch_goal_pos)]
      batch_finished |= np.array(at_goal, dtype=bool)
      if np.all(batch_finished):
          break

  return finished
