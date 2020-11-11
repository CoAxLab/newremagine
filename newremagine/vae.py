__all__ = ['VAE', '_loss_function', 'train', 'test']

from torchvision.datasets import FashionMNIST
import numpy as np

import torch
import torch as th
import torch.optim as optim
import torch.nn.functional as F
from torch import nn

from torch.utils.tensorboard import SummaryWriter


class VAE(nn.Module):
    """A classic VAE.

    Params
    ------
    input_dim : int
        The size of the (flattened) image vector 
    latent_dim : int
        The size of the latent memory    
    """
    def __init__(self, input_dim=784, latent_dim=20):
        super(VAE, self).__init__()
        # Set dims
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        # Init the layers in the deep net
        self.fc1 = nn.Linear(self.input_dim, 400)
        self.fc21 = nn.Linear(400, self.latent_dim)
        self.fc22 = nn.Linear(400, self.latent_dim)
        self.fc3 = nn.Linear(self.latent_dim, 400)
        self.fc4 = nn.Linear(400, self.input_dim)

    def encode(self, x):
        """Encode a torch tensor (batch_size, inpiut_size)"""
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Expand a latent memory, to input_size."""
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def sample(self, n, device=None):
        """Use noise to sample n images from latent space."""
        with torch.no_grad():
            x = torch.randn(n, self.latent_dim)
            x = x.to(device)
            samples = self.decode(x)
            return samples

    def forward(self, x):
        """Get a reconstructed image"""
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self._reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def _loss_function(recon_x, x, mu, logvar, input_dim):
    """Reconstruction + KL divergence losses summed over all elements and batch"""
    BCE = F.binary_cross_entropy(recon_x,
                                 x.view(-1, input_dim),
                                 reduction="sum")

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(train_batch, model, optimizer, device, input_dim):
    """A single VAE training step"""

    model.train()
    batch = train_batch.to(device)
    optimizer.zero_grad()
    recon_batch, mu, logvar = model(train_batch)
    loss = _loss_function(recon_batch, train_batch, mu, logvar, input_dim)
    loss.backward()
    optimizer.step()
    return loss


def test(test_data, model, device, input_dim):
    """Test a VAE on a whole dataset"""

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_data):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += _loss_function(recon_batch, data, mu, logvar,
                                        input_dim).item()

    return test_loss