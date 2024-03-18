import torch
import numpy as np
from pathlib import Path

def get_device():
  return (
    torch.device('cuda') if torch.cuda.is_available() else
    torch.device('mps') if torch.backends.mps.is_available() else
    torch.device('cpu'))

def save_checkpoint(model, checkpoint_fn):
  checkpoint_dir = Path(__file__).parent / 'checkpoints'
  checkpoint_dir.mkdir(exist_ok=True)
  torch.save(model.state_dict(), checkpoint_dir / checkpoint_fn)

def batch(data, indices=None, device=torch.device('cpu')):
  if indices is not None:
    data = [data[i] for i in indices]
  return tuple(torch.as_tensor(np.array(x), device=device) for x in zip(*data))