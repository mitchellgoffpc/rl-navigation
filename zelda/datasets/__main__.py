import random
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from collections import Counter
from torch.utils.data import DataLoader
from zelda.datasets import ImitationDataset
from zelda.datasets.generate import generate_random_episodes, generate_expert_episodes

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest='command')
  visualize_parser = subparsers.add_parser('visualize')
  inspect_parser = subparsers.add_parser('inspect')
  benchmark_parser = subparsers.add_parser('benchmark')
  generate_parser = subparsers.add_parser('generate')
  generate_parser.add_argument('type', choices=['expert', 'random'])
  args = parser.parse_args()

  # Visualize data
  if args.command == 'inspect':
    map_positions = []
    dataset = ImitationDataset('expert')
    for i in trange(len(dataset), desc="inspecting dataset"):
      *_, info = dataset[i]
      map_positions.append(tuple(info['map_pos'].tolist()))
    print("ROOM COUNT:")
    for pos, count in Counter(map_positions).items():
      print(f"{pos}: {count}")

  elif args.command == 'visualize':
    dataset = ImitationDataset('random', max_goal_dist=1)
    # dataset = ValidationDataset()
    for i in range(5):
      image, goal_state, action, info = random.choice(dataset)
      fig, ax = plt.subplots(1, 2, figsize=(10, 5))
      ax[0].imshow(image.numpy())
      ax[0].set_title(f'State\nAction: {action}')
      ax[1].imshow(goal_state.numpy())
      ax[1].set_title(f'Goal State\nDistance: {info["distance"]}, map_pos: {info["map_pos"].tolist()}, screen_pos: {info["screen_pos"].tolist()}')
      plt.tight_layout()
      plt.show()

  # Benchmark dataset
  elif args.command == 'benchmark':
    dataset = ImitationDataset('random')
    for i in trange(1000, desc="benchmarking dataset"):
      dataset[i]

    # Benchmark dataloader
    dataloader = DataLoader(dataset, batch_size=32, num_workers=8, shuffle=True)
    for _ in tqdm(dataloader, desc="benchmarking dataloader, n=8, bs=32"):
      pass

  elif args.command == 'generate':
    if args.type == 'random':
      generate_random_episodes()
    elif sys.type == "expert":
      generate_expert_episodes()
