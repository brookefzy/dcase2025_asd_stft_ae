import torch
from torch import nn

class AENet(nn.Module):
    def __init__(self,input_dim, block_size):
        super(AENet,self).__init__()
        self.input_dim = input_dim
        self.cov_source = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        self.cov_target = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim,128),
            nn.BatchNorm1d(128,momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.BatchNorm1d(128,momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.BatchNorm1d(128,momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.BatchNorm1d(128,momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128,8),
            nn.BatchNorm1d(8,momentum=0.01, eps=1e-03),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(8,128),
            nn.BatchNorm1d(128,momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.BatchNorm1d(128,momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.BatchNorm1d(128,momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.BatchNorm1d(128,momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128,self.input_dim)
        )

    def forward(self, x):
        z = self.encoder(x.view(-1,self.input_dim))
        return self.decoder(z), z


class VAENet(nn.Module):
    """Feed forward variational autoencoder used for the baseline."""

    def __init__(self, input_dim, block_size, latent_dim=8):
        super(VAENet, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.cov_source = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        self.cov_target = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, self.input_dim),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x.view(-1, self.input_dim))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
