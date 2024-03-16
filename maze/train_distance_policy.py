import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import trange
from maze.models import MLP
from maze.environment import MazeEnv
from maze.helpers import batch, generate_data, evaluate_policy

NUM_EPISODES = 10000
NUM_STEPS = 10
BATCH_SIZE = 128

def train():
  env = MazeEnv(7, 7)
  device = torch.device('cpu')
  distance_model = MLP(math.prod(env.observation_space.shape), env.action_space.n).to(device)
  distance_model.load_state_dict(torch.load(Path(__file__).parent / 'checkpoints/distance.ckpt'))
  policy_model = MLP(math.prod(env.observation_space.shape), env.action_space.n).to(device)
  policy_optimizer = torch.optim.AdamW(policy_model.parameters(), lr=3e-4)
  train_steps = generate_data(env, NUM_EPISODES, NUM_STEPS, split="train")
  val_steps = generate_data(env, NUM_EPISODES, NUM_STEPS, split="val")

  for _ in range(5):

    # training
    train_loss = 0
    train_correct, train_total = 0, 0
    policy_model.train()
    random.shuffle(train_steps)
    for i in trange(0, len(train_steps), BATCH_SIZE, leave=False, desc="DISTANCE POLICY - TRAIN"):
      states, goals, actions, _, correct_actions, _, _ = batch(train_steps[i:i+BATCH_SIZE])
      with torch.no_grad():
        distances = distance_model(torch.as_tensor(states), torch.as_tensor(goals))
        targets = F.softmax(-(distances - distances.mean()), dim=-1)

      preds = policy_model(torch.as_tensor(states), torch.as_tensor(goals))
      loss = F.cross_entropy(preds, targets, reduction="none")
      policy_optimizer.zero_grad()
      loss.mean().backward()
      policy_optimizer.step()
      train_loss += loss.sum().item()
      train_correct += np.sum(np.argmax(preds.cpu().detach().numpy(), axis=-1) == correct_actions)
      train_total += len(states)

    # validation
    val_loss = 0
    val_correct, val_total = 0, 0
    policy_model.eval()
    with torch.no_grad():
      for i in trange(0, len(val_steps), BATCH_SIZE, leave=False, desc="DISTANCE POLICY - VAL"):
        states, goals, actions, _, correct_actions, correct_distances, correct_next_distances = batch(train_steps[i:i+BATCH_SIZE])
        with torch.no_grad():
          distances = distance_model(torch.as_tensor(states), torch.as_tensor(goals))
          targets = distances.argmin(dim=-1)

        preds = policy_model(torch.as_tensor(states), torch.as_tensor(goals))
        loss = F.cross_entropy(preds, targets, reduction="none")
        val_loss += loss.sum().item()
        val_correct += np.sum(np.argmax(preds.cpu().numpy(), axis=-1) == correct_actions)
        val_total += len(states)

    def policy_fn(state, goal):
      with torch.no_grad():
        probs = F.softmax(policy_model(torch.as_tensor([state]), torch.as_tensor([goal])), dim=-1)
        return torch.multinomial(probs, 1).item()
    win_rate = evaluate_policy(env, policy_fn, 1000, 20)

    # print metrics
    torch.save(policy_model.state_dict(), Path(__file__).parent / 'checkpoints/policy-db.ckpt')
    print(f"DISTANCE POLICY: "
          f"Train accuracy: {train_correct / train_total * 100:.2f}%, Train loss: {train_loss / train_total:.4f}, "
          f"Val accuracy: {val_correct / val_total * 100:.2f}%, Val loss: {val_loss / val_total:.4f}, "
          f"Win rate: {win_rate * 100:.2f}%")


if __name__ == '__main__':
  train()
