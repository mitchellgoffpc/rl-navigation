import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from zelda.models import ZeldaAgent
from zelda.datasets import RLDataset

NUM_EPOCHS = 20
BATCH_SIZE = 32

def train():
    device = torch.device('cuda')
    model = ZeldaAgent().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    dataset = RLDataset('random')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    for epoch in range(NUM_EPOCHS):
        for batch in (pbar := tqdm(dataloader)):
            state, goal_state, action, next_state = (item.to(device) for item in batch[:-1]) # skip info

            predicted_distances = model(state, goal_state)
            predicted_distances = predicted_distances[range(len(state)), action]
            with torch.no_grad():
                target_distances = model(next_state, goal_state)
            done = (next_state == goal_state).all(3).all(2).all(1)
            target_distances = 1 + (1-done.float()) * target_distances.min(dim=-1).values

            loss = F.mse_loss(predicted_distances, target_distances)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # calculate accuracy
            correct_predictions = predicted_distances.argmax(dim=-1) == action
            accuracy = correct_predictions.sum().item() / BATCH_SIZE
            pbar.set_description(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Loss: {loss.item():.3f} | Accuracy: {accuracy*100:.2f}")


if __name__ == '__main__':
    train()
