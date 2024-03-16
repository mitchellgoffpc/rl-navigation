import math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import trange
from maze.models import MLP
from maze.environment import MazeEnv
from maze.helpers import batch, generate_data

NUM_EPISODES = 10000
NUM_STEPS = 10
BATCH_SIZE = 128

def train():
  env = MazeEnv(7, 7)
  device = torch.device('cpu')
  distance_model = MLP(math.prod(env.observation_space.shape), env.action_space.n).to(device)
  distance_optimizer = torch.optim.AdamW(distance_model.parameters(), lr=3e-4)
  train_steps = generate_data(env, NUM_EPISODES, NUM_STEPS, split="train")
  val_steps = generate_data(env, NUM_EPISODES, NUM_STEPS, split="val")

  for _ in range(5):

    # training
    train_loss = 0
    train_total = 0
    distance_model.train()
    for i in trange(0, len(train_steps), BATCH_SIZE, leave=False, desc="DISTANCE - TRAIN"):
      episode_idxs = np.random.randint(0, NUM_EPISODES, size=BATCH_SIZE)
      start_idxs = np.random.randint(0, NUM_STEPS - 1, size=BATCH_SIZE)
      state_idxs = episode_idxs * NUM_STEPS + start_idxs # np.random.randint(0, NUM_STEPS, size=BATCH_SIZE)
      goal_idxs = episode_idxs * NUM_STEPS + np.random.randint(start_idxs + 1, NUM_STEPS, size=BATCH_SIZE)
      states, _, actions, *_ = batch(train_steps, state_idxs)
      goals, *_ = batch(train_steps, goal_idxs)
      step_diffs = goal_idxs - state_idxs # np.abs(state_idxs - goal_idxs)

      preds = distance_model(torch.as_tensor(states), torch.as_tensor(goals))
      preds = preds[torch.arange(len(actions)), actions]
      loss = F.mse_loss(preds.squeeze(), torch.as_tensor(step_diffs, dtype=torch.float32).to(device), reduction="none")
      distance_optimizer.zero_grad()
      loss.mean().backward()
      distance_optimizer.step()
      train_loss += loss.sum().item()
      train_total += len(states)

    # validation
    val_loss = 0
    val_total = 0
    distance_model.eval()
    with torch.no_grad():
      for i in trange(0, len(val_steps), BATCH_SIZE, leave=False, desc="DISTANCE - VAL"):
        episode_idxs = np.random.randint(0, NUM_EPISODES, size=BATCH_SIZE)
        start_idxs = np.random.randint(0, NUM_STEPS - 1, size=BATCH_SIZE)
        state_idxs = episode_idxs * NUM_STEPS + start_idxs # np.random.randint(0, NUM_STEPS, size=BATCH_SIZE)
        goal_idxs = episode_idxs * NUM_STEPS + np.random.randint(start_idxs + 1, NUM_STEPS, size=BATCH_SIZE)
        states, _, actions, *_ = batch(val_steps, state_idxs)
        goals, *_ = batch(val_steps, goal_idxs)
        step_diffs = goal_idxs - state_idxs # np.abs(state_idxs - goal_idxs)

        preds = distance_model(torch.as_tensor(states), torch.as_tensor(goals))
        preds = preds[torch.arange(len(actions)), actions]
        loss = F.mse_loss(preds.squeeze(), torch.as_tensor(step_diffs, dtype=torch.float32).to(device), reduction="sum")
        val_loss += loss.item()
        val_total += len(states)

    # print metrics
    torch.save(distance_model.state_dict(), Path(__file__).parent / 'checkpoints/distance.ckpt')
    print(f"DISTANCE: "
          f"Train loss: {train_loss / train_total:.4f}, "
          f"Val loss: {val_loss / val_total:.4f}")


if __name__ == "__main__":
  train()