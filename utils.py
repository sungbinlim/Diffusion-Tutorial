import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vision_utils
import seaborn as sns
import pandas as pd
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
    file_name = title + '_generated_images.png'
    plt.savefig(fname=file_name)

def save_image_seqs(batch, title):
    row = batch[0].shape[0]
    col = len(batch)
    fig = plt.figure(figsize=(row, col))
    for i, img_seq in enumerate(batch):
        ax = fig.add_subplot(col, 1, i+1)
        plot_batch(ax, img_seq.cpu())
    file_name = title + '_generated_sequential_images.png'
    plt.savefig(fname=file_name)

def plot_seqs(imgs, with_orig=False, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(figsize=(200,200), nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [image] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()

# The Fourier transform function
def spectrum(sample):  
    f = torch.fft.fft2(sample, norm='ortho')
    fshift = torch.fft.fftshift(f) 
    magnitude_spectrum = torch.log(torch.abs(fshift))
    
    return magnitude_spectrum

def average_psd(tensor):
    a = torch.fft.rfft(tensor, axis=0)
    a = a.real ** 2 + a.imag ** 2
    a = torch.sum(a, axis=1) / a.shape[1]
    f = torch.fft.rfftfreq(tensor[0].shape[0])
    return f.numpy(), a.numpy()

def plot_psds(imgs, time):
    sns.set_style('ticks')
    for (i, psd_image) in enumerate(imgs):
        data = {'frequency': psd_image[0],
                'amplitude': psd_image[1]
                }
        df = pd.DataFrame(data)
        sns.lineplot(x='frequency', y='amplitude', data=df, label=f'noise level at: {time[i]}')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    sns.despine()
    plt.show()

if __name__ == "__main__":
    x = torch.randn(size=(5, 3, 64, 64))
    y = torch.randn(size=(5, 3, 64, 64))
    img_list = [x, y]
    save_image_seqs(img_list, "test")