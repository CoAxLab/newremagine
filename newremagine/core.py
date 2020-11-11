__all__ = ['train', 'test', 'plot_latent', 'plot_test']

from torchvision.datasets import FashionMNIST
import numpy as np

import torch
import torch as th
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from newremagine import vae
from newremagine import replay

import matplotlib.pyplot as plt
from copy import deepcopy


def train(
    fraction,
    train_dataset,
    num_episodes=10,
    batch_size=10,
    num_burn=1,
    lr=0.001,
    device="cpu",
    perfect=True,
    replay_name="Replay",
    replay_kwargs=None,
    vae_name="VAE",
    vae_kwargs=None,
):
    """Given a fraction and dataset, train to self-supervised model
    
    Params
    -----
    fraction : 3-tuple
        A set of three probability values, setting the probability of 
        sampling new data, replaying old data, or imagining data and
        trainong the data.
    train_dataset : a torch dataset object
        The data to train on
    num_epsidoes : int
        The fixed number of training trials
    batch_size : int
        The size of batches to use when traning the network
    num_burn : int (> 0)
        The number of episodes before the we try and replar or 
        imagine data. Both these need a min number of experiences
        before they could be useful. 
    lr : float (> 0)
        The learning rate of the network
    device : str
        The device to use for training. Either 'cpu` or 'cuda:0'.
        See torch docs for more on this.
    perfect : bool
        If True, replay uses exact copies of the data. If False
        replay uses reconstructed data from the network itself
    replay_name : str
        Any name of a classe found in the `newremainge.replay` module
        can be used here.
    replay_kwargs : dict
        Keword arguments to be transparently passed the the Replay
        memory 
    replay_name : str
        Any name of a classe found in the `newremainge.vae` module
        can be used here.
    replay_kwargs : dict
        Keword arguments to be transparently passed the the VAE
        network
    """

    # -- Init memories
    if replay_kwargs is None:
        replay_kwargs = {}
    if vae_kwargs is None:
        vae_kwargs = {}

    Replay = getattr(replay, replay_name)
    memory = Replay(**replay_kwargs)

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
    options = ["new", "replay", "imagine"]
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
                train_dataset[i][0]
                for i in range(batch_idx, batch_idx + batch_size)
            ]
            train_batch = torch.stack(train_batch)
            batch_idx += batch_size
            # If the memory is perfect we replay the
            # training data later, but If the memory
            # is imperfect we add the reconstructed data
            # the the memory intead.
            if perfect:
                memory.encode(train_batch)
            else:
                with torch.no_grad():
                    recon_batch, _, _ = model(train_batch)
                    memory.encode(recon_batch)
        elif option == "replay":
            train_batch = memory.sample(1)
        else:
            # Sampling from the VAE model is a kind of
            # imagination, or so we imagine in here
            train_batch = model.sample(batch_size, device=device)

        # Train the vae/model
        loss = vae.train(train_batch, model, optimizer, device,
                         model.input_dim)

    return model, memory, float(loss)


def test(model, test_dataset, device="cpu"):
    """Test a pre-trained model on a new dataset.
    
    Params
    -----
    model : torch nn.Module instance
        The model we want to test. 
    test_dataset : a torch dataset object
        The data to train on
     device : str
        The device to use for training. Either 'cpu` or 'cuda:0'.
        See torch docs for more on this
    """
    return vae.test(test_dataset, model, device, model.input_dim)


def plot_latent(model, n, img_size=28):
    """Display a grid of samples from the latent space.

    Params
    -----
    model : torch nn.Module instance
        The model we want to test.
    n : int
        The size of the grid
    img_size : int
        The size of images in the orginal data (We assume they
        are greyscale)
    """

    # Make a display grid
    figure = np.zeros((img_size * n, img_size * n))

    # MAke data
    x = model.sample(n**2)
    x = x.flatten().numpy().reshape(n**2, img_size, img_size)
    imgs = [x[i, ::] for i in range(n**2)]

    # !
    k = 0
    for i in range(n):
        for j in range(n):
            figure[i * img_size:(i + 1) * img_size,
                   j * img_size:(j + 1) * img_size] = imgs[k]
            k += 1

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap="gray")
    plt.axis("off")
    plt.show()


def plot_test(test_dataset, model, n, img_size=28):
    """Display a grid random samples from the test data.
    
    Params
    -----
    model : torch nn.Module instance
        The model we want to test.
    test_dataset : a torch dataset object
        The data to train on
    n : int
        The size of the grid
    img_size : int
        The size of images in the orginal data (We assume they
        are greyscale)
    """

    # Make a display grid
    figure = np.zeros((img_size * n, img_size * n))

    # Choose data
    idx = np.random.randint(0, len(test_dataset), size=n**2)

    # !
    imgs = []
    for i in idx:
        with torch.no_grad():
            x, _ = test_dataset[i]
            x_reconst, _, _ = model(x)

        img = x_reconst.unsqueeze(0).numpy().flatten().reshape(
            img_size, img_size)
        imgs.append(deepcopy(img))

    k = 0
    for i in range(n):
        for j in range(n):
            figure[i * img_size:(i + 1) * img_size,
                   j * img_size:(j + 1) * img_size] = imgs[k]
            k += 1

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap="gray")
    plt.axis("off")
    plt.show()