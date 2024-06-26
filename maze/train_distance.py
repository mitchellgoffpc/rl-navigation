import math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import trange
from common.helpers import batch
from maze.models import MLP
from maze.environment import MazeEnv
from maze.helpers import generate_data, evaluate_policy

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
    for _ in trange(0, len(train_steps), BATCH_SIZE, leave=False, desc="DISTANCE - TRAIN"):
      state_idxs = np.random.randint(0, len(train_steps) - 1, size=BATCH_SIZE)
      states, _, actions, *_, steps_left, success = batch(train_steps, state_idxs, device=device)
      goal_idxs = np.random.randint(state_idxs, state_idxs + steps_left.numpy() + 1, size=BATCH_SIZE)
      goal_idxs = np.where(steps_left.numpy() == 0, state_idxs, goal_idxs)  # Since numpy doesn't support randint with low == high
      goals, *_ = batch(train_steps, goal_idxs)
      step_diffs = goal_idxs - state_idxs

      diffs = (states != goals).sum(dim=[1, 2])  # Assuming states and goals are 4D (batch, channel, height, width)
      assert torch.all(diffs <= 2), "States and goals differ in more than two pixels per pair."
      assert np.all(step_diffs >= 0)

      preds = distance_model(states, goals)
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
      for _ in trange(0, len(val_steps), BATCH_SIZE, leave=False, desc="DISTANCE - VAL"):
        state_idxs = np.random.randint(0, len(train_steps) - 1, size=BATCH_SIZE)
        states, _, actions, *_, steps_left, success = batch(train_steps, state_idxs, device=device)
        goal_idxs = np.random.randint(state_idxs, state_idxs + steps_left.numpy() + 1, size=BATCH_SIZE)
        goal_idxs = np.where(steps_left.numpy() == 0, state_idxs, goal_idxs)  # Since numpy doesn't support randint with low == high
        goals, *_ = batch(train_steps, goal_idxs)
        step_diffs = goal_idxs - state_idxs

        preds = distance_model(states, goals)
        preds = preds[torch.arange(len(actions)), actions]
        loss = F.mse_loss(preds.squeeze(), torch.as_tensor(step_diffs, dtype=torch.float32).to(device), reduction="sum")
        val_loss += loss.item()
        val_total += len(states)

    def policy_fn(state, goal):
      with torch.no_grad():
        distances = distance_model(state, goal)
        probs = F.softmax(-(distances - distances.mean()), dim=-1)
        return torch.multinomial(probs, 1).item()
    win_rate = evaluate_policy(env, policy_fn, 1000, 20, device=device)

    # print metrics
    torch.save(distance_model.state_dict(), Path(__file__).parent / 'checkpoints/distance.ckpt')
    print(f"DISTANCE: "
          f"Train loss: {train_loss / train_total:.4f}, "
          f"Val loss: {val_loss / val_total:.4f}, "
          f"Win rate: {win_rate * 100:.2f}%")


if __name__ == "__main__":
  train()