import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vision_utils
from data import reverse_transform



def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)

def plot_batch(ax, batch, title=None, **kwargs):
    r_transform = reverse_transform()
    imgs = vision_utils.make_grid(batch, padding=2, normalize=True)
    imgs = r_transform(imgs)
    ax.set_axis_off()
    if title is not None: ax.set_title(title)
    return ax.imshow(imgs, **kwargs)

def save_images(batch, title):
    batch_size = batch.shape[0]
    row = int(np.sqrt(batch_size))
    col = batch_size // row
    fig = plt.figure(figsize=(row, col))
    ax = fig.add_subplot(111)
    plot_batch(ax, batch.cpu(), title)
    file_name = title + '_generated images.png'
    plt.savefig(fname=file_name)

def save_image_seqs(batch, title):
    row = batch[0].shape[0]
    col = len(batch)
    fig = plt.figure(figsize=(row, col))
    for i, img_seq in enumerate(batch):
        ax = fig.add_subplot(col, 1, i+1)
        plot_batch(ax, img_seq.cpu())
    file_name = title + '_generated sequential images.png'
    plt.savefig(fname=file_name)