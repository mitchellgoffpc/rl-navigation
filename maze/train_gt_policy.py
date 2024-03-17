import math
import random
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
    for i in trange(0, len(train_steps), BATCH_SIZE, leave=False, desc="GT POLICY - TRAIN"):
      states, goals, _, _, _, correct_actions, *_ = batch(train_steps[i:i+BATCH_SIZE], device=device)
      preds = policy_model(states, goals)
      loss = F.cross_entropy(preds, correct_actions, reduction="none")
      policy_optimizer.zero_grad()
      loss.mean().backward()
      policy_optimizer.step()

      train_loss += loss.sum().item()
      train_correct += torch.sum(preds.argmax(dim=-1) == correct_actions).item()
      train_total += len(states)

    # validation
    val_loss = 0
    val_correct, val_total = 0, 0
    policy_model.eval()
    with torch.no_grad():
      for i in trange(0, len(val_steps), BATCH_SIZE, leave=False, desc="GT POLICY - VAL"):
        states, goals, _, _, _, correct_actions, *_ = batch(val_steps[i:i+BATCH_SIZE], device=device)
        preds = policy_model(states, goals)
        loss = F.cross_entropy(preds, correct_actions, reduction="sum")
        val_loss += loss.item()
        val_correct += torch.sum(preds.argmax(dim=-1) == correct_actions).item()
        val_total += len(states)

    def policy_fn(state, goal):
      with torch.no_grad():
        probs = F.softmax(policy_model(state, goal), dim=-1)
        return torch.multinomial(probs, 1).item()
    win_rate = evaluate_policy(env, policy_fn, 1000, 20, device=device)

    # print metrics
    torch.save(policy_model.state_dict(), Path(__file__).parent / 'checkpoints/gt-policy.ckpt')
    print(f"GT POLICY: "
          f"Train accuracy: {train_correct / train_total * 100:.2f}%, Train loss: {train_loss / train_total:.4f}, "
          f"Val accuracy: {val_correct / val_total * 100:.2f}%, Val loss: {val_loss / val_total:.4f}, "
          f"Win rate: {win_rate * 100:.2f}%")


if __name__ == "__main__":
  train()
