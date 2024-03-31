import sys
import math
import torch
import torch.nn.functional as F
from pathlib import Path
from maze.models import MLP
from maze.environment import MazeEnv
from maze.helpers import evaluate_policy

if __name__ == '__main__':
  assert len(sys.argv) == 2, "Usage: python evaluate.py <model_name>"
  model_name = sys.argv[1]
  path = Path(__file__).parent / 'checkpoints' / f'{model_name}.ckpt'
  assert path.exists(), f"Checkpoint {path} not found"

  env = MazeEnv(7, 7)
  policy_model = MLP(math.prod(env.observation_space.shape), env.action_space.n)
  policy_model.load_state_dict(torch.load(path))
  policy_model.eval()

  def policy_fn(state, goal):
    with torch.no_grad():
      probs = F.softmax(policy_model(state, goal), dim=-1)
      return torch.multinomial(probs, 1).item()
  win_rate = evaluate_policy(env, policy_fn, 1000, 20)
  print(f"Win rate: {win_rate*100:.2f}%")
