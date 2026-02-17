"""Training utilities and simple loop helpers."""

from typing import Callable, Iterable


def train_epoch(model, data_loader: Iterable, loss_fn: Callable, opt):
    total_loss = 0.0
    for x, y in data_loader:
        pred = model(x)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss
