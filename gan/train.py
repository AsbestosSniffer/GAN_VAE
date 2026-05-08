from glob import glob
import os
import torch
from tqdm import tqdm
from utils import get_fid, interpolate_latent_space, save_plot
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from torchvision.datasets import VisionDataset


def build_transforms():
    # Convert input to tensor and rescale [0,1] -> [-1,1]
    rescaling = lambda x: (x - 0.5) * 2.0
    ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])
    return ds_transforms


def get_optimizers_and_schedulers(gen, disc):
    optim_discriminator = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.0, 0.9))
    optim_generator = torch.optim.Adam(gen.parameters(), lr=2e-4, betas=(0.0, 0.9))
    # Disc LR decays to 0 over 500K steps; gen LR decays to 0 over 100K steps
    scheduler_discriminator = torch.optim.lr_scheduler.LambdaLR(
        optim_discriminator, lambda step: max(0.0, 1.0 - step / 500_000)
    )
    scheduler_generator = torch.optim.lr_scheduler.LambdaLR(
        optim_generator, lambda step: max(0.0, 1.0 - step / 100_000)
    )
    return (
        optim_discriminator,
        scheduler_discriminator,
        optim_generator,
        scheduler_generator,
    )


class Dataset(VisionDataset):
    def __init__(self, root, transform=None):
        super(Dataset, self).__init__(root)
        self.file_names = glob(os.path.join(self.root, "*.jpg"), recursive=True)
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.file_names[index])
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.file_names)


def train_model(
    gen,
    disc,
    num_iterations,
    batch_size,
    lamb=10,
    prefix=None,
    gen_loss_fn=None,
    disc_loss_fn=None,
    log_period=10000,
    amp_enabled=True,
):
    torch.backends.cudnn.benchmark = True
    ds_transforms = build_transforms()
    train_loader = torch.utils.data.DataLoader(
        Dataset(root="../datasets/CUB_200_2011_32", transform=ds_transforms),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    (
        optim_discriminator,
        scheduler_discriminator,
        optim_generator,
        scheduler_generator,
    ) = get_optimizers_and_schedulers(gen, disc)

    scaler = torch.cuda.amp.GradScaler()

    iters = 0
    fids_list = []
    iters_list = []
    pbar = tqdm(total=num_iterations)
    while iters < num_iterations:
        for train_batch in train_loader:
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                train_batch = train_batch.cuda()

                # ---- Update Discriminator ----
                fake_batch = gen(n_samples=train_batch.shape[0])
                discrim_real = disc(train_batch)
                discrim_fake = disc(fake_batch.detach())

                # Interpolated batch for WGAN-GP gradient penalty
                alpha = torch.rand(train_batch.shape[0], 1, 1, 1, device=train_batch.device)
                interp = (alpha * train_batch + (1 - alpha) * fake_batch.detach()).requires_grad_(True)
                discrim_interp = disc(interp)

            discriminator_loss = disc_loss_fn(
                discrim_real, discrim_fake, discrim_interp, interp, lamb
            )

            optim_discriminator.zero_grad(set_to_none=True)
            scaler.scale(discriminator_loss).backward()
            scaler.step(optim_discriminator)
            scheduler_discriminator.step()

            if iters % 5 == 0:
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    fake_batch = gen(n_samples=train_batch.shape[0])
                    discrim_fake = disc(fake_batch)
                    generator_loss = gen_loss_fn(discrim_fake)

                optim_generator.zero_grad(set_to_none=True)
                scaler.scale(generator_loss).backward()
                scaler.step(optim_generator)
                scheduler_generator.step()

            if iters % log_period == 0 and iters != 0:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        # generate 100 samples in [0, 1]
                        generated_samples = (gen(n_samples=100) / 2 + 0.5).clamp(0, 1)
                    save_image(
                        generated_samples.data.float(),
                        prefix + "samples_{}.png".format(iters),
                        nrow=10,
                    )
                    if os.environ.get('PYTORCH_JIT', 1):
                        torch.jit.save(torch.jit.script(gen), prefix + "/generator.pt")
                        torch.jit.save(torch.jit.script(disc), prefix + "/discriminator.pt")
                    else:
                        torch.save(gen, prefix + "/generator.pt")
                        torch.save(disc, prefix + "/discriminator.pt")
                    fid = get_fid(
                        gen,
                        dataset_name="cub",
                        dataset_resolution=32,
                        z_dimension=128,
                        batch_size=256,
                        num_gen=10_000,
                    )
                    print(f"Iteration {iters} FID: {fid}")
                    fids_list.append(fid)
                    iters_list.append(iters)

                    save_plot(
                        iters_list,
                        fids_list,
                        xlabel="Iterations",
                        ylabel="FID",
                        title="FID vs Iterations",
                        filename=prefix + "fid_vs_iterations",
                    )
                    interpolate_latent_space(
                        gen, prefix + "interpolations_{}.png".format(iters)
                    )
            scaler.update()
            iters += 1
            pbar.update(1)
    fid = get_fid(
        gen,
        dataset_name="cub",
        dataset_resolution=32,
        z_dimension=128,
        batch_size=256,
        num_gen=50_000,
    )
    print(f"Final FID (Full 50K): {fid}")
