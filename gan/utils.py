import argparse
import torch
from cleanfid import fid
from matplotlib import pyplot as plt
from torchvision.utils import save_image


def save_plot(x, y, xlabel, ylabel, title, filename):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename + ".png")


@torch.no_grad()
def get_fid(gen, dataset_name, dataset_resolution, z_dimension, batch_size, num_gen):
    gen_fn = lambda z: (gen.forward_given_samples(z) / 2 + 0.5) * 255
    score = fid.compute_fid(
        gen=gen_fn,
        dataset_name=dataset_name,
        dataset_res=dataset_resolution,
        num_gen=num_gen,
        z_dim=z_dimension,
        batch_size=batch_size,
        verbose=True,
        dataset_split="custom",
    )
    return score


@torch.no_grad()
def interpolate_latent_space(gen, path):
    # 10x10 grid: dim0 and dim1 linearly interpolated in [-1, 1], rest zero
    z = torch.zeros(100, 128, device=next(gen.parameters()).device)
    vals = torch.linspace(-1, 1, 10)
    for i, v1 in enumerate(vals):
        for j, v2 in enumerate(vals):
            z[i * 10 + j, 0] = v1
            z[i * 10 + j, 1] = v2
    samples = gen.forward_given_samples(z)
    samples = (samples / 2 + 0.5).clamp(0, 1)
    save_image(samples, path, nrow=10)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable_amp", action="store_true")
    args = parser.parse_args()
    return args
