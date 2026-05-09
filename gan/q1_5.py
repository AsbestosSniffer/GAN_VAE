import os

import torch
from utils import get_args

from networks import Discriminator, Generator
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    """
    WGAN-GP discriminator loss.
    loss = E[D(fake)] - E[D(real)] + lambda * E[(||grad D(interp)|| - 1)^2]
    """
    loss_pt1 = discrim_fake.mean() - discrim_real.mean()

    # Gradient penalty
    gradients = torch.autograd.grad(
        outputs=discrim_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(discrim_interp),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient_norm = gradients.reshape(gradients.shape[0], -1).norm(2, dim=1)
    loss_pt2 = lamb * ((gradient_norm - 1) ** 2).mean()

    loss = loss_pt1 + loss_pt2
    return loss


def compute_generator_loss(discrim_fake):
    # WGAN-GP generator loss: -E[D(fake)]
    loss = -discrim_fake.mean()
    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_wgan_gp/"
    os.makedirs(prefix, exist_ok=True)

    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=5000,
        amp_enabled=not args.disable_amp,
    )
