# AUTOGENERATED! DO NOT EDIT! File to edit: core.ipynb (unless otherwise specified).

__all__ = ['train', 'test']

# Cell
from torchvision.datasets import FashionMNIST
import numpy as np

import torch
import torch as th
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

# Cell
from newremagine import vae
from newremagine import recall

# Cell
# `newremagine`
def train(
    fraction,
    train_dataset,
    num_episodes=10,
    batch_size=10,
    num_burn=1,
    lr=0.001,
    device="cpu",
    perfect=True,
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

    # -- Everything is sane?
    num_episodes = int(num_episodes)
    num_burn = int(num_burn)
    lr = float(lr)
    if not np.isclose(sum(fraction), 1):
        raise ValueError("fractions must sum to 1")

    # -- !
    options = ["new", "recall", "imagine"]
    batch_idx = 0
    for n in range(num_episodes):
        # Burn in the model, or consider other options?
        if n < num_burn:
            option = "new"
        else:
            option = np.random.choice(options, p=fraction)

        # Make/get training data
        if option == "new":
            # Get new data
            train_batch = [
                train_dataset[i][0] for i in range(batch_idx, batch_idx + batch_size)
            ]
            train_batch = torch.stack(train_batch)
            batch_idx += batch_size
            # ....
            # If the memory is perfect we recall the training data later
            # If the memory is imperfect we add the reconstructed data
            if perfect:
                memory.encode(train_batch)
            else:
                recon_batch, _, _ = model(train_batch)
                memory.encode(recon_batch)

        elif option == "recall":
            train_batch = memory.sample(1)
        else:
            # Sampling from the VAE model is a kind of
            # imagination, or so we imagine in here
            train_batch = model.sample(batch_size, device=device)

        # Train the vae/model
        loss = vae.train(train_batch, model, optimizer, device, model.input_dim)

    return model, memory, float(loss)

# Cell
def test(model, test_dataset, device="cpu"):
    return vae.test(test_dataset, model, device, model.input_dim)