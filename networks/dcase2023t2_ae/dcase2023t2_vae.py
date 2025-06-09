from networks.dcase2023t2_ae.dcase2023t2_ae import DCASE2023T2AE
from networks.dcase2023t2_ae.network import VAENet
import torch.nn.functional as F

class DCASE2023T2VAE(DCASE2023T2AE):
    """Variational AutoEncoder model."""

    def init_model(self):
        self.block_size = self.data.height
        return VAENet(input_dim=self.data.input_dim, block_size=self.block_size)

    def loss_fn(self, recon_x, x, mu, logvar):
        recon = F.mse_loss(recon_x, x.view(recon_x.shape), reduction="none")
        recon = torch.mean(recon, dim=1)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl = kl / self.data.input_dim
        return recon + kl
