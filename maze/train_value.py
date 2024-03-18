import math
import random
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
ALPHA = 0.998  # Polyak averaging coefficient

def train():
  env = MazeEnv(7, 7)
  device = torch.device('cpu')
  value_model = MLP(math.prod(env.observation_space.shape), env.action_space.n).to(device)
  target_model = MLP(math.prod(env.observation_space.shape), env.action_space.n).to(device)
  target_model.load_state_dict(value_model.state_dict())
  value_optimizer = torch.optim.AdamW(value_model.parameters(), lr=3e-4)
  train_steps = generate_data(env, NUM_EPISODES, NUM_STEPS, split="train")
  val_steps = generate_data(env, NUM_EPISODES, NUM_STEPS, split="val")

  for _ in range(15):

    # training
    train_loss = 0
    train_total = 0
    train_correct = 0
    value_model.train()
    random.shuffle(train_steps)
    for i in trange(0, len(train_steps), BATCH_SIZE, leave=False, desc="VALUE - TRAIN"):
      states, goals, actions, next_states, done, correct_actions, *_ = batch(train_steps[i:i+BATCH_SIZE])
      with torch.no_grad():
        next_values = target_model(next_states, goals).min(dim=1).values * ~done

      values = value_model(states, goals)
      action_values = values[torch.arange(len(actions)), actions]
      loss = F.mse_loss(action_values, next_values + 1, reduction="none")
      value_optimizer.zero_grad()
      loss.mean().backward()
      value_optimizer.step()
      train_loss += loss.sum().item()
      train_total += len(states)
      train_correct += (values.argmin(dim=1) == correct_actions).sum().item()

      for target_param, param in zip(target_model.parameters(), value_model.parameters()):
        target_param.data.copy_(target_param.data * ALPHA + param.data * (1 - ALPHA))

    # validation
    val_loss = 0
    val_total = 0
    val_correct = 0
    value_model.eval()
    with torch.no_grad():
      for i in trange(0, len(val_steps), BATCH_SIZE, leave=False, desc="VALUE - VAL"):
        states, goals, actions, next_states, done, correct_actions, *_ = batch(train_steps[i:i+BATCH_SIZE])
        next_values = value_model(next_states, goals).min(dim=1).values * ~done
        values = value_model(states, goals)
        action_values = values[torch.arange(len(actions)), actions]
        loss = F.mse_loss(action_values, next_values + 1, reduction="none")
        val_loss += loss.sum().item()
        val_total += len(states)
        val_correct += (values.argmin(dim=1) == correct_actions).sum().item()

    def policy_fn(state, goal):
      with torch.no_grad():
        distances = value_model(state, goal)
        return distances.argmin().item()
    win_rate = evaluate_policy(env, policy_fn, 1000, 20)

    # print metrics
    torch.save(value_model.state_dict(), Path(__file__).parent / 'checkpoints/value.ckpt')
    print(f"VALUE: "
          f"Train loss: {train_loss / train_total:.4f}, Train accuracy: {train_correct / train_total * 100:.2f}%, "
          f"Val loss: {val_loss / val_total:.4f}, Val accuracy: {val_correct / val_total * 100:.2f}%, "
          f"Win rate: {win_rate * 100:.2f}%")


if __name__ == "__main__":
  train()



