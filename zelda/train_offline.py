import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from common.helpers import get_device
from zelda.models import ResNet
from zelda.datasets import ImitationDataset, RLDataset, ValidationDataset

NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
ALPHA = 0.995  # for target model update

device = get_device()
value_model = ResNet(4).to(device)
target_model = ResNet(4).to(device)
target_model.load_state_dict(value_model.state_dict())
optimizer = torch.optim.AdamW(value_model.parameters(), lr=LEARNING_RATE)

train_dataset = RLDataset('random', subsample=0.1, max_goal_dist=2)
test_dataset = RLDataset('random', max_episode_len=10, max_goal_dist=1)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

for epoch in range(NUM_EPOCHS):

  # training
  value_model.train()
  for batch in (pbar := tqdm(train_dataloader)):
    state, goal_state, action, next_state = (item.to(device) for item in batch[:-1]) # skip info
    values = value_model(state, goal_state)
    action_values = values[range(len(state)), action]
    done = (next_state == goal_state).all(3).all(2).all(1)
    with torch.no_grad():
      future_distances = target_model(next_state, goal_state).min(dim=-1).values * ~done

    loss = F.mse_loss(action_values, future_distances + 1)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    correct_predictions = values.argmin(dim=-1) == action
    accuracy = correct_predictions.sum().item() / BATCH_SIZE
    pbar.set_description(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Loss: {loss.item():.3f} | Accuracy: {accuracy*100:.2f}%")

    for param, target_param in zip(value_model.parameters(), target_model.parameters()):
        target_param.data.copy_(ALPHA * target_param.data + (1 - ALPHA) * param.data)

  # validation
  value_model.eval()
  correct_predictions = 0
  total_predictions = 0
  for batch in (pbar := tqdm(test_dataloader)):
    state, goal_state, action, _ = (item.to(device) for item in batch[:-1]) # skip info
    with torch.no_grad():
      values = target_model(state, goal_state)
    correct_predictions += (values.argmin(dim=-1) == action).sum().item()
    total_predictions += action.shape[0]
  val_accuracy = correct_predictions / total_predictions
  print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Val Accuracy: {val_accuracy*100:.2f}%")


if __name__ == '__main__':
  train()
