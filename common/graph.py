import numpy as np
from typing import List, Tuple, Optional, Callable

KeyFunction = Callable[[Tuple[np.ndarray]], int]
DistanceFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]
ActionFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]
Action = int

class ReplayGraph:
  size: int
  states: List[np.ndarray]
  episode: List[Tuple[np.ndarray, ...]]
  episodes: List[Tuple[np.ndarray, ...]]
  weights: Optional[np.ndarray]
  distances: Optional[np.ndarray]

  def __init__(self, size:int):
    self.size = size
    self.states = []
    self.episode = []
    self.episodes = []
    self.weights = None
    self.distances = None

  def __len__(self) -> int:
    return len(self.distances) if self.distances is not None else 0

  def add_step(self, *data:np.ndarray):
    self.episode.append(data)

  def add_episode(self, *data:np.ndarray):
    self.episodes.append(data)

  def commit(self):
    assert len(self.episode) > 0, "Can't commit an empty episode!"
    columns = []
    for i in range(len(self.episode[0])):
      columns.append([step[i] for step in self.episode])
    self.add_episode(*columns)
    self.episode = []

  def compile(self, dist:DistanceFunction, key:KeyFunction = id):
    # Create a list of nodes and edges
    self.nodes, self.edges = {}, {}
    for episode in self.episodes:
      states, _, actions, *_ = episode
      for i in range(len(states)):
        self.nodes[key(states[i])] = states[i]
      for i in range(len(states)-1):
        self.edges[(key(states[i]), key(states[i+1]))] = (states[i], states[i+1], actions[i])

    size = len(self.nodes)
    states = list(self.nodes.values())
    node_indices = {k:i for i,k in enumerate(self.nodes.keys())}

    # Create the weight matrix
    self.weights = np.full((size, size), np.inf, dtype=np.float32)
    self.weights[np.arange(size), np.arange(size)] = 0
    for skey, nskey in self.edges.keys():
      self.weights[node_indices[skey], node_indices[nskey]] = 1

    # Evaluate the model to fill in the rest of the edges
    for i in range(size):
      indices = np.where(self.weights[i] == np.inf)[0]
      batch_states = np.repeat(states[i][None], len(indices), axis=0)
      batch_goals = np.stack([states[j] for j in indices], axis=0)
      distances = dist(batch_states, batch_goals)
      self.weights[i,indices] = distances

    # Floyd-Warshall to compute distances between edges
    self.distances = self.weights.copy()
    for k in range(size):
      for i in range(size):
        self.distances[i] = np.minimum(self.distances[i], self.distances[i,k] + self.distances[k])

  def search(self, state:np.ndarray, goal:np.ndarray, dist:DistanceFunction, act:ActionFunction) -> Action:
    size = len(self.distances)
    distances = np.full((size, size), np.inf, dtype=np.float32)
    distances[:size,:size] = self.distances

    # Fill in the distances from each node to the goal (with Floyd-Warshall update)
    batch_states = np.repeat(state[None], size, axis=0)
    batch_goals = np.stack(list(self.nodes.values()), axis=0)
    distances[-1] = dist(batch_states, batch_goals)

    for k in range(size):
      for i in range(size):
        distances[i,-1] = min(distances[i,-1], distances[i,k] + distances[k,-1])

    # Fill in the distances from the state to each node
    batch_states = np.stack(list(self.nodes.values()), axis=0)
    batch_goals = np.repeat(goal[None], size, axis=0)
    distances[:,-1] = dist(batch_states, batch_goals)
    distances[-1,-1] = np.inf

    # Pick the best intermediate goal and select an action
    node_idx = np.argmin(distances[-1] + distances[:,-1])
    action = act(state[None], list(self.nodes.values())[node_idx][None]).squeeze()
    return act(state[None], goal[None])

  def sample(self, bs:int) -> Tuple[np.ndarray, ...]:
    assert self.distances is not None, "You must call graph.compile() before you can sample from it"
    nodes, edges = list(self.nodes.values()), list(self.edges.values())
    node_keys, edge_keys = list(self.nodes.keys()), list(self.edges.keys())
    node_indices = {k:i for i,k in enumerate(self.nodes.keys())}

    edge_idxs = np.random.randint(0, len(edges), size=bs)
    goal_idxs = np.random.randint(0, len(nodes), size=bs)
    batch_edge_keys = [edge_keys[i] for i in edge_idxs]
    batch_goal_keys = [node_keys[i] for i in goal_idxs]
    batch_edges = [edges[i] for i in edge_idxs]
    batch_goals = [nodes[i] for i in goal_idxs]

    states = np.array([s for s, _, _ in batch_edges])
    actions = np.array([a for _, _, a in batch_edges])
    goals = np.array(batch_goals)
    targets = np.array([
      self.distances[node_indices[sk], node_indices[nsk]] + \
      self.distances[node_indices[nsk], node_indices[gk]]
      for (sk, nsk), gk
      in zip(batch_edge_keys, batch_goal_keys)])

    return states, goals, actions, targets


# TESTING

if __name__ == '__main__':
  import time

  graph = ReplayGraph(0)
  for _ in range(10):
    states = np.random.randint(0, 10, size=(100, 3))
    graph.add_episode(states)

  # graph.compile(lambda x, y: 100)
  # print(graph.weights)
  # print(graph.distances)

  print("Benchmarking...")
  num_samples = 10
  st = time.perf_counter()
  for _ in range(num_samples):
    graph.compile(lambda x, y: 100)
  et = time.perf_counter()
  print(f"Benchmark: {(et-st)/num_samples*1000:.2f}ms/sample")
