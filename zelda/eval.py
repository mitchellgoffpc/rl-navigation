import gym
import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from pathlib import Path
from zelda.models import ZeldaAgent
from zelda.environment import ZeldaEnvironment

BS = 8
NUM_STEPS = 20
NUM_REPEATS = 8
NUM_ROLLOUTS = 128

def main(plot=True):
    device = (
        torch.device('cuda') if torch.cuda.is_available() else
        torch.device('mps') if torch.backends.mps.is_available() else
        torch.device('cpu'))

    env = gym.vector.make('zelda', BS)
    policy = ZeldaAgent(4).eval().to(device)
    policy.load_state_dict(torch.load(Path(__file__).parent / 'checkpoints' / 'policy.ckpt'))
    torch.set_grad_enabled(False)

    # Run rollouts to collect candidate goal states
    goals, goal_positions = [], []
    for _ in trange(0, NUM_ROLLOUTS, BS, desc="Generating goal states"):
        ep_states, ep_positions = [], []
        obs, _ = env.reset()
        for _ in range(NUM_STEPS):
            action = env.action_space.sample()
            for _ in range(NUM_REPEATS):
                obs, _, _, _, info = env.step(action)
            ep_states.append(obs)
            ep_positions.append(info['pos'])

        for i, idx in enumerate(np.random.randint(0, len(ep_states), size=BS)):
          goals.append(ep_states[idx][i])
          goal_positions.append(ep_positions[idx][i])

    # Run the model in a loop, one episode per goal state
    finished = np.zeros(NUM_ROLLOUTS, dtype=bool)
    for offset in trange(0, NUM_ROLLOUTS, BS, desc="Evaluating model"):
        obs, _ = env.reset()
        batch_goals = np.stack(goals[offset:offset+BS], axis=0)
        batch_goal_pos = goal_positions[offset:offset+BS]
        batch_finished = finished[offset:offset+BS]

        for _ in range(NUM_STEPS):
            with torch.no_grad():
                action_probs = policy(obs, batch_goals).softmax(-1)
            action = torch.multinomial(action_probs, 1).cpu().numpy().squeeze(1)
            for _ in range(NUM_REPEATS):
                obs, _, _, _, info = env.step(action)
            at_goal = [ZeldaEnvironment.pos_matches(pos, goal_pos, tolerance=10) for pos, goal_pos in zip(info['pos'], batch_goal_pos)]
            batch_finished |= np.array(at_goal, dtype=bool)
            if np.all(batch_finished):
                break

    print(f"Success rate: {np.mean(finished) * 100}%")

    # Plot which goals were successfully reached
    if args.plot:
        goal_positions = np.array(goal_positions, dtype=int)
        plt.scatter(goal_positions[finished][:, 0], 255 - goal_positions[finished][:, 1])
        plt.scatter(goal_positions[~finished][:, 0], 255 - goal_positions[~finished][:, 1])
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a policy model')
    parser.add_argument('--plot', action='store_true', help='Plot goal states')
    args = parser.parse_args()

    main(plot=args.plot)
