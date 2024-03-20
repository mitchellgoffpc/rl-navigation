import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from common.helpers import get_device
from zelda.models import ResNet
from zelda.datasets import ZeldaDataset
from zelda.helpers import correct_actions

NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
ALPHA = 0.990  # for target model update

device = get_device()
value_model = ResNet(4).to(device)
target_model = ResNet(4).to(device)
target_model.load_state_dict(value_model.state_dict())
optimizer = torch.optim.AdamW(value_model.parameters(), lr=LEARNING_RATE)

train_dataset = ZeldaDataset('random', subsample=0.1)
test_dataset = ZeldaDataset('random', subsample=0.01, max_episode_len=10)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

for epoch in range(NUM_EPOCHS):

  # training
  value_model.train()
  train_correct, train_total = 0, 0
  for batch in (pbar := tqdm(train_dataloader, leave=False)):
    state, goal_state, action, next_state = (item.to(device) for item in batch[:-1]) # skip info
    pos, goal_pos = batch[-1]['pos'], batch[-1]['goal_pos']

    values = value_model(state, goal_state)
    action_values = values[range(len(state)), action]
    done = (pos - goal_pos).abs().sum(dim=-1) < 10
    with torch.no_grad():
      future_distances = target_model(next_state, goal_state).min(dim=-1).values * ~done.to(device)

    loss = F.l1_loss(action_values, future_distances + 1)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    predicted_actions = values.argmin(dim=-1).cpu()
    correct_action_mask = correct_actions(pos, goal_pos)
    train_correct += correct_action_mask[torch.arange(len(action)), predicted_actions].sum().item()
    train_total += len(state)
    pbar.set_description(f"TRAIN | Epoch {epoch + 1}/{NUM_EPOCHS} | Loss: {loss.item():.3f} | Accuracy: {train_correct/train_total*100:.2f}%")

    for param, target_param in zip(value_model.parameters(), target_model.parameters()):
      target_param.data.copy_(ALPHA * target_param.data + (1 - ALPHA) * param.data)

  # validation
  value_model.eval()
  val_correct, val_total = 0, 0
  for batch in (pbar := tqdm(test_dataloader, leave=False)):
    state, goal_state, action, _ = (item.to(device) for item in batch[:-1]) # skip info
    with torch.no_grad():
      values = target_model(state, goal_state)
    predicted_actions = values.argmin(dim=-1).cpu()
    correct_action_mask = correct_actions(batch[-1]['pos'], batch[-1]['goal_pos'])
    val_correct += correct_action_mask[torch.arange(len(action)), predicted_actions].sum().item()
    val_total += len(state)
    pbar.set_description(f"VAL | Epoch {epoch + 1}/{NUM_EPOCHS} | Loss: {loss.item():.3f} | Accuracy: {train_correct/train_total*100:.2f}%")

  print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Accuracy: {train_correct/train_total*100:.2f}% | Val Accuracy: {val_correct/val_total*100:.2f}%")


if __name__ == '__main__':
  train()
