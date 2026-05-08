import torch
import torch.nn as nn
import torch.nn.functional as F


class UpSampleConv2D(torch.jit.ScriptModule):
    def __init__(
        self,
        input_channels,
        kernel_size=3,
        n_filters=128,
        upscale_factor=2,
        padding=0,
    ):
        super(UpSampleConv2D, self).__init__()
        self.conv = nn.Conv2d(
            input_channels, n_filters, kernel_size=kernel_size, padding=padding
        )
        self.upscale_factor = upscale_factor

    @torch.jit.script_method
    def forward(self, x):
        # Nearest-neighbor upsampling via pixel shuffle:
        # repeat channels upscale_factor^2 times, then PixelShuffle
        x = x.repeat(1, self.upscale_factor * self.upscale_factor, 1, 1)
        x = torch.pixel_shuffle(x, self.upscale_factor)
        return self.conv(x)


class DownSampleConv2D(torch.jit.ScriptModule):
    def __init__(
        self, input_channels, kernel_size=3, n_filters=128, downscale_ratio=2, padding=0
    ):
        super(DownSampleConv2D, self).__init__()
        self.conv = nn.Conv2d(
            input_channels, n_filters, kernel_size=kernel_size, padding=padding
        )
        self.downscale_ratio = downscale_ratio

    @torch.jit.script_method
    def forward(self, x):
        # Spatial mean pooling via pixel unshuffle then average
        x = torch.pixel_unshuffle(x, self.downscale_ratio)
        # x: (B, C*r^2, H/r, W/r)
        B, Cr2, H, W = x.shape
        r2 = self.downscale_ratio * self.downscale_ratio
        C = Cr2 // r2
        x = x.view(B, r2, C, H, W)      # (B, r^2, C, H/r, W/r)
        x = x.permute(1, 0, 2, 3, 4)    # (r^2, B, C, H/r, W/r)
        x = x.mean(dim=0)               # (B, C, H/r, W/r)
        return self.conv(x)


class ResBlockUp(torch.jit.ScriptModule):
    """
    ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(in_channels, ...)
            (1): ReLU()
            (2): Conv2d(in_channels, n_filters, kernel_size=(3,3), padding=(1,1), bias=False)
            (3): BatchNorm2d(n_filters, ...)
            (4): ReLU()
            (5): UpSampleConv2D(conv: Conv2d(n_filters, n_filters, kernel_size=(3,3), padding=(1,1)))
        )
        (upsample_residual): UpSampleConv2D(conv: Conv2d(input_channels, n_filters, kernel_size=(1,1)))
    )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockUp, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(input_channels),
            nn.ReLU(),
            nn.Conv2d(input_channels, n_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            UpSampleConv2D(n_filters, kernel_size=3, n_filters=n_filters, padding=1),
        )
        self.upsample_residual = UpSampleConv2D(
            input_channels, kernel_size=1, n_filters=n_filters, padding=0
        )

    @torch.jit.script_method
    def forward(self, x):
        return self.layers(x) + self.upsample_residual(x)


class ResBlockDown(torch.jit.ScriptModule):
    """
    ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3,3), padding=(1,1))
            (2): ReLU()
            (3): DownSampleConv2D(conv: Conv2d(n_filters, n_filters, kernel_size=(3,3), padding=(1,1)))
        )
        (downsample_residual): DownSampleConv2D(conv: Conv2d(input_channels, n_filters, kernel_size=(1,1)))
    )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockDown, self).__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(input_channels, n_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            DownSampleConv2D(n_filters, kernel_size=3, n_filters=n_filters, padding=1),
        )
        self.downsample_residual = DownSampleConv2D(
            input_channels, kernel_size=1, n_filters=n_filters, padding=0
        )

    @torch.jit.script_method
    def forward(self, x):
        return self.layers(x) + self.downsample_residual(x)


class ResBlock(torch.jit.ScriptModule):
    """
    ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3,3), padding=(1,1))
            (2): ReLU()
            (3): Conv2d(n_filters, n_filters, kernel_size=(3,3), padding=(1,1))
        )
    )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(input_channels, n_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
        )

    @torch.jit.script_method
    def forward(self, x):
        return self.layers(x) + x


class Generator(torch.jit.ScriptModule):
    """
    Generator(
      (dense): Linear(128, 2048)
      (layers): Sequential(
        (0): ResBlockUp(128->128)
        (1): ResBlockUp(128->128)
        (2): ResBlockUp(128->128)
        (3): BatchNorm2d(128)
        (4): ReLU()
        (5): Conv2d(128, 3, 3, padding=1)
        (6): Tanh()
      )
    )
    """

    def __init__(self, starting_image_size=4):
        super(Generator, self).__init__()
        self.dense = nn.Linear(128, 128 * starting_image_size * starting_image_size)
        self.starting_image_size = starting_image_size
        self.layers = nn.Sequential(
            ResBlockUp(128, n_filters=128),
            ResBlockUp(128, n_filters=128),
            ResBlockUp(128, n_filters=128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    @torch.jit.script_method
    def forward_given_samples(self, z):
        x = self.dense(z)
        x = x.view(-1, 128, self.starting_image_size, self.starting_image_size)
        return self.layers(x)

    @torch.jit.script_method
    def forward(self, n_samples: int = 1024):
        z = torch.randn(n_samples, 128, device=self.dense.weight.device)
        return self.forward_given_samples(z)


class Discriminator(torch.jit.ScriptModule):
    """
    Discriminator(
      (layers): Sequential(
        (0): ResBlockDown(3->128)
        (1): ResBlockDown(128->128)
        (2): ResBlock(128->128)
        (3): ResBlock(128->128)
        (4): ReLU()
      )
      (dense): Linear(128, 1)
    )
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            ResBlockDown(3, n_filters=128),
            ResBlockDown(128, n_filters=128),
            ResBlock(128, n_filters=128),
            ResBlock(128, n_filters=128),
            nn.ReLU(),
        )
        self.dense = nn.Linear(128, 1)

    @torch.jit.script_method
    def forward(self, x):
        x = self.layers(x)
        # sum over spatial dimensions
        x = x.sum(dim=[2, 3])
        return self.dense(x)
