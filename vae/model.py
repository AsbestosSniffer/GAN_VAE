import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim


class Encoder(nn.Module):
    """
    Sequential(
        (0): Conv2d(3, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        (1): ReLU()
        (2): Conv2d(32, 64, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        (3): ReLU()
        (4): Conv2d(64, 128, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        (5): ReLU()
        (6): Conv2d(128, 256, kernel_size=(3,3), stride=(2,2), padding=(1,1))
    )
    """

    def __init__(self, input_shape, latent_dim):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.convs = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        )
        # After 3 stride-2 convs on H x W image: H//8 x W//8
        H, W = input_shape[1], input_shape[2]
        self.fc = nn.Linear(256 * (H // 8) * (W // 8), latent_dim)

    def forward(self, x):
        x = self.convs(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)


class VAEEncoder(Encoder):
    def __init__(self, input_shape, latent_dim):
        super().__init__(input_shape, latent_dim)
        # Override fc to output 2*latent_dim (mu and log_std)
        H, W = input_shape[1], input_shape[2]
        self.fc = nn.Linear(256 * (H // 8) * (W // 8), 2 * latent_dim)

    def forward(self, x):
        x = self.convs(x)
        x = x.reshape(x.shape[0], -1)
        out = self.fc(x)
        mu = out[:, : self.latent_dim]
        log_std = out[:, self.latent_dim :]
        return mu, log_std


class Decoder(nn.Module):
    """
    Sequential(
        (0): ReLU()
        (1): ConvTranspose2d(256, 128, kernel_size=(4,4), stride=(2,2), padding=(1,1))
        (2): ReLU()
        (3): ConvTranspose2d(128, 64, kernel_size=(4,4), stride=(2,2), padding=(1,1))
        (4): ReLU()
        (5): ConvTranspose2d(64, 32, kernel_size=(4,4), stride=(2,2), padding=(1,1))
        (6): ReLU()
        (7): Conv2d(32, 3, kernel_size=(3,3), stride=(1,1), padding=(1,1))
    )
    """

    def __init__(self, latent_dim, output_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape

        H, W = output_shape[1], output_shape[2]
        self.base_size = (256, H // 8, W // 8)
        self.fc = nn.Linear(latent_dim, 256 * (H // 8) * (W // 8))
        self.deconvs = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.reshape(-1, *self.base_size)
        return self.deconvs(x)


class AEModel(nn.Module):
    def __init__(self, variational, latent_size, input_shape=(3, 32, 32)):
        super().__init__()
        assert len(input_shape) == 3

        self.input_shape = input_shape
        self.latent_size = latent_size
        if variational:
            self.encoder = VAEEncoder(input_shape, latent_size)
        else:
            self.encoder = Encoder(input_shape, latent_size)
        self.decoder = Decoder(latent_size, input_shape)
    # NOTE: call model.encoder and model.decoder directly in train.py
