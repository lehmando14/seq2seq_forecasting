import torch
from torch import nn
from torch.utils import data

from . import gru_model as gm

def train_one_epoch(
        dataloader: data.DataLoader, model: gm.GRU, loss_fn, optimizer,
        label_offset: int, label_size: int
    ) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    size = len(dataloader.dataset)
    model.set_label_specs(label_offset, label_size)    
    model.to(device)
    model.train()

    total_loss = 0.0
    batch_count = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        pred = pred.reshape((-1, model.max_transactions + 1))
        y = y.reshape((-1, model.max_transactions + 1))
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Accumulate the loss
        total_loss += loss.item()
        batch_count += 1

        if batch % 101 == 0:
            average_loss = total_loss / batch_count
            current = (batch + 1) * len(X)
            print(f"Average loss: {average_loss:>7f}  [{current:>5d}/{size:>5d}]")
            
            # Reset total loss and batch count
            total_loss = 0.0
            batch_count = 0

def test(dataloader, model, loss_fn):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)

            pred = pred.reshape((-1, model.max_transactions + 1))
            y = y.reshape((-1, model.max_transactions + 1))
            test_loss += loss_fn(pred, y).item()
    
    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")