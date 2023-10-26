import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from zelda.models import ZeldaAgent
from zelda.datasets import ImitationDataset, RLDataset, ValidationDataset

NUM_EPOCHS = 20
BATCH_SIZE = 32

def train():
    device = (
      torch.device('cuda') if torch.cuda.is_available() else
      torch.device('mps') if torch.backends.mps.is_available() else
      torch.device('cpu'))

    model = ZeldaAgent().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    train_dataset = RLDataset('random', subsample=0.1, max_goal_dist=1)
    test_dataset = ImitationDataset('expert', max_episode_len=10, max_goal_dist=1)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch in (pbar := tqdm(train_dataloader)):
            state, goal_state, action, next_state = (item.to(device) for item in batch[:-1]) # skip info

            predicted_distances = model(state, goal_state)
            predicted_distances = predicted_distances[range(len(state)), action]
            with torch.no_grad():
                best_target_distances = model(next_state, goal_state).min(dim=-1).values
            done = (next_state == goal_state).all(3).all(2).all(1)
            target_distances = 1 + best_target_distances * ~done
            # print(predicted_distances.mean().item(), target_distances.mean().item(), done.float().mean().item())

            loss = F.mse_loss(predicted_distances, target_distances)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            correct_predictions = predicted_distances.argmax(dim=-1) == action
            accuracy = correct_predictions.sum().item() / BATCH_SIZE
            pbar.set_description(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Loss: {loss.item():.3f} | Accuracy: {accuracy*100:.2f}")

        model.eval()
        correct_predictions = 0
        total_predictions = 0
        for batch in (pbar := tqdm(test_dataloader)):
            state, goal_state, action = (item.to(device) for item in batch[:-1]) # skip info
            with torch.no_grad():
                predicted_distances = model(state, goal_state)
            correct_predictions += (predicted_distances.argmax(dim=-1) == action).sum().item()
            total_predictions += action.shape[0]
        val_accuracy = correct_predictions / total_predictions
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Val Accuracy: {val_accuracy*100:.2f}")


if __name__ == '__main__':
    train()
