import os

import torch
import torch.nn.functional as F
from utils import get_args

from networks import Discriminator, Generator
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    # LSGAN discriminator: eq.(2) with a=0, b=1
    loss = 0.5 * ((discrim_real - 1) ** 2).mean() + 0.5 * (discrim_fake ** 2).mean()
    return loss


def compute_generator_loss(discrim_fake):
    # LSGAN generator: eq.(2) with c=1
    loss = 0.5 * ((discrim_fake - 1) ** 2).mean()
    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_ls_gan/"
    os.makedirs(prefix, exist_ok=True)

    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )
