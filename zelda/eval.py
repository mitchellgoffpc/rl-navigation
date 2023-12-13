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
NUM_REPEATS = 8
NUM_ROLLOUTS = 128

def main(plot=True):
    env = gym.vector.make('zelda', BS)
    policy = ZeldaAgent(4).eval()
    policy.load_state_dict(torch.load(Path(__file__).parent / 'checkpoints' / 'policy.ckpt'))
    torch.set_grad_enabled(False)

    # Run rollouts to collect candidate goal states
    rollouts = []
    for _ in trange(NUM_ROLLOUTS // BS, desc="Generating goal states"):
        states = []
        positions = []
        obs, _ = env.reset()
        for _ in range(20):
            action = env.action_space.sample()
            for _ in range(NUM_REPEATS):
                obs, _, _, _, info = env.step(action)
            states.append(obs)
            positions.append(info['pos'])
        for i in range(BS):
            rollouts.append([(state[i], pos[i]) for state, pos in zip(states, positions)])

    # Select 100 goal states, randomly sampled from these rollouts
    goal_states = [random.choice(rollout) for rollout in rollouts]

    # Plot the goal states
    if args.plot:
        goal_positions = np.array(goal_states, dtype=int)
        plt.scatter(goal_positions[:, 0], 255 - goal_positions[:, 1])
        plt.show()

    # Run the model in a loop, one episode per goal state
    # batches = [goal_states[i:i+BS] for i in range(0, len(goal_states), BS)]
    env = gym.make('zelda')
    successes = 0
    for goal, goal_pos in tqdm(goal_states, desc="Evaluating model"):
        obs, _ = env.reset()
        for _ in range(20):
            action_probs = policy(obs[None], goal[None]).softmax(-1)
            action = torch.multinomial(action_probs, 1).cpu().numpy().item()
            for _ in range(NUM_REPEATS):
                obs, _, _, _, info = env.step(action)
            if env.pos_matches(info['pos'], goal_pos):
                successes += 1
                break

    # Record the percentage of rollouts which successfully reach the goal state
    success_rate = successes / len(goal_states)
    print(f"Success rate: {success_rate * 100}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a policy model')
    parser.add_argument('--plot', action='store_true', help='Plot goal states')
    args = parser.parse_args()

    main(plot=args.plot)
