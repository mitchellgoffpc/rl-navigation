import time
import numpy as np
from enum import Enum
from typing import List, Dict, Any

class MetricType(Enum):
  SUM = 'SUM'
  MEAN = 'MEAN'

class Metrics:
  metrics: Dict[str, List[Any]]

  def __init__(self, metric_types:Dict[str, MetricType]):
    self.metric_types = metric_types
    self.reset()

  def reset(self):
    self.last_report_time = time.time()
    self.metrics = {}

  def add(self, data:Dict[str, Any]):
    for k,v in data.items():
      if k not in self.metrics:
        self.metrics[k] = []
      self.metrics[k].append(v)

  def report(self, n):
    metrics = [f"Episode {n:<6}"]
    for k,v in self.metrics.items():
      if self.metric_types.get(k) is MetricType.SUM:
        metrics.append("{k}: {v:<4}".format(k=k, v=np.sum(v)))
      else:
        metrics.append("{k}: {v:.2f}".format(k=k, v=np.mean(v)))
    metrics.append(f"Time: {time.time() - self.last_report_time:.2f}s")
    print(" | ".join(metrics))
    self.reset()
