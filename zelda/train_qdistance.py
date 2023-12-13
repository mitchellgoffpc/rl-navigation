import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from common.replay import ReplayBuffer
from zelda.models import ZeldaAgent
from zelda.environment import ZeldaEnvironment

BATCH_SIZE = 32
NUM_STEPS = 20
NUM_REPEATS = 8
NUM_EPISODES = 100
LEARNING_RATE = 3e-4

def train():
    device = (
      torch.device('cuda') if torch.cuda.is_available() else
      torch.device('mps') if torch.backends.mps.is_available() else
      torch.device('cpu'))

    env = ZeldaEnvironment()
    replay = ReplayBuffer(100)
    model = ZeldaAgent(4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for _ in range(NUM_EPISODES):

        # generate random rollout
        model.eval()
        for _ in range(NUM_STEPS):
            action = env.action_space.sample()
            for _ in range(NUM_REPEATS):
                obs, _, _, _, info = env.step(action)
            replay.add_step(obs, action, info['pos'])
        replay.commit()

        # train on samples from replay buffer
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

        # save models
        checkpoint_dir = Path(__file__).parent / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), checkpoint_dir / 'qdistance.ckpt')


if __name__ == '__main__':
    train()
