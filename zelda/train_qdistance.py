import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from common.replay import ReplayBuffer
from zelda.models import ZeldaAgent
from zelda.environment import ZeldaEnvironment
from zelda.helpers import get_device, save_checkpoint, generate_rollout, generate_goals, evaluate_model

BATCH_SIZE = 128
NUM_STEPS = 20
NUM_REPEATS = 8
NUM_EPISODES = 10000
NUM_TRAINING_STEPS = 1
LEARNING_RATE = 3e-4

def train():
  device = get_device()
  env = ZeldaEnvironment()
  replay = ReplayBuffer(100)
  model = ZeldaAgent(4).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

  for ep in range(NUM_EPISODES):

    # generate random rollout
    model.eval()
    for obs, action, info in generate_rollout(env, NUM_STEPS):
      replay.add_step(obs, action, info['pos'])
    replay.commit()

    # train on samples from replay buffer
    model.train()
    for _ in range(NUM_TRAINING_STEPS):
      states, actions, _, mask = replay.sample(BATCH_SIZE, NUM_STEPS)
      targets = np.random.randint(0, np.sum(mask, axis=-1), size=BATCH_SIZE)
      goals = states[np.arange(BATCH_SIZE), targets]

      with torch.no_grad():
        best_future_distances = model(states[:,1], goals).min(dim=-1).values.cpu() * torch.as_tensor(targets > 1)
      distance_preds = model(states[:,0], goals).cpu()[torch.arange(len(actions)), actions[:,0]]
      loss = F.smooth_l1_loss(distance_preds, best_future_distances + 1)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

    def policy(obs, goal):
      return model(obs, goal).argmin(dim=-1)

    if ep % 10 == 0:
      goal_states, goal_positions = generate_goals(env, num_goals=16, num_steps=NUM_STEPS)
      eval_results = evaluate_model(env, policy, goal_states, goal_positions, NUM_STEPS)
      print(eval_results.mean())

    save_checkpoint(model, 'qdistance.ckpt')


if __name__ == '__main__':
  train()
