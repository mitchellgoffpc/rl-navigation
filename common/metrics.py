import numpy as np
from enum import Enum
from typing import List, Dict, Any

class MetricType(Enum):
  SUM = 'SUM'
  MEAN = 'MEAN'

class Metrics:
  report_interval: int
  num_updates: int
  metrics: Dict[str, List[Any]]

  def __init__(self, report_interval:int, metric_types:Dict[str, MetricType]):
    self.report_interval = report_interval
    self.metric_types = metric_types
    self.num_updates = 0
    self.reset()

  def reset(self):
    self.metrics = {}

  def add(self, data:Dict[str, Any]):
    for k,v in data.items():
      if k not in self.metrics:
        self.metrics[k] = []
      self.metrics[k].append(v)

    self.num_updates += 1
    if self.num_updates % self.report_interval == 0:
      self.report()
      self.reset()

  def report(self):
    metrics = [f"Episode {self.num_updates:<6}"]
    for k,v in self.metrics.items():
      if self.metric_types.get(k) is MetricType.SUM:
        metrics.append("{k}: {v:<4}".format(k=k, v=np.sum(v)))
      else:
        metrics.append("{k}: {v:.2f}".format(k=k, v=np.mean(v)))
    print(" | ".join(metrics))
