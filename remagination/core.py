# AUTOGENERATED! DO NOT EDIT! File to edit: exp.ipynb (unless otherwise specified).

__all__ = ['train_fashion', 'test_fashion']

# Cell
from torchvision.datasets import FashionMNIST
import numpy as np

import torch
import torch as th
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

# Cell
from remagination import vae
from remagination import recall

# Cell
def train_fashion(
    fraction,
    num_episodes=10,
    batch_size=10,
    num_burn=1,
    lr=0.001,
    device="cpu",
    recall_name="Recall",
    recall_kwargs=None,
    vae_name="VAE",
    vae_kwargs=None,
):

    # -- Init memories
    if recall_kwargs is None:
        recall_kwargs = {}
    if vae_kwargs is None:
        vae_kwargs = {}

    Recall = getattr(recall, recall_name)
    memory = Recall(**recall_kwargs)

    VAE = getattr(vae, vae_name)
    model = VAE(**vae_kwargs).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # -- Get the data (assumes data/ exists)
    dataset = FashionMNIST("data/", train=True, download=True)

    # -- Everything is sane?
    num_episodes = int(num_episodes)
    num_burn = int(num_burn)
    lr = float(lr)
    if not np.isclose(sum(fraction), 1):
        raise ValueError("fractions must sum to 1")
    if num_episodes > (len(dataset) % batch_size):
        raise ValueError(f"num_episodes must be <= {len(dataset) % batch_size}")

    # -- !
    options = ["new", "recall", "imagine"]
    batch_idx = 0
    for n in num_episodes:
        # Burn in the model, or consider other options?
        if n < num_burn:
            option = "new"
        else:
            option = np.random.choice(options, p=fraction)

        # Make/get training data
        if option == "new":
            train_batch = dataset[idx : batch_idx + batch_size]
            memory.update(train_batch)
            batch_idx += batch_size
        elif option == "recall":
            train_batch = memory.sample(batch_size)
        else:
            train_batch = model.sample(batch_size, device=device)

        # Train the vae/model
        loss = vae.train(train_batch, model, optimizer, device)

    return vae, loss

# Cell
def test_fashion(model, device="cpu"):
    test_dataset = FashionMNIST("data/", train=False)
    return vae.test(test_dataset, model, device)