import numpy as np
import torch

# Training code
def make_train_step(model, loss_fn, optimizer):
    def train_step_fn(x, t, y):

        model.train()
        y_hat = model(x, t)
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss.item()
    return train_step_fn

# Evaluation code
def make_valid_step(model, loss_fn):
    def valid_sten_fn(x, t, y):

        model.eval()
        y_hat = model(x, t)
        loss = loss_fn(y_hat, y)
        
        return loss.item()
    return valid_sten_fn