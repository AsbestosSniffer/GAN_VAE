import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import (
    cosine_beta_schedule,
    default,
    extract,
    unnormalize_to_zero_to_one,
)
from einops import rearrange, reduce


class DiffusionModel(nn.Module):
    def __init__(
        self,
        model,
        timesteps=1000,
        sampling_timesteps=None,
        ddim_sampling_eta=1.,
    ):
        super(DiffusionModel, self).__init__()

        self.model = model
        self.channels = self.model.channels
        self.device = torch.cuda.current_device()

        self.betas = cosine_beta_schedule(timesteps).to(self.device)
        self.num_timesteps = self.betas.shape[0]

        alphas = 1. - self.betas
        # Cumulative products: alpha_bar_t = prod_{i=1}^{t} alpha_i
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        # alpha_bar_{t-1}: prepend 1.0 for t=0 case
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Coefficient of x_t when predicting x_0: 1 / sqrt(alpha_bar_t)
        self.x_0_pred_coef_1 = 1.0 / torch.sqrt(self.alphas_cumprod)
        # Coefficient of pred_noise when predicting x_0: -sqrt(1 - alpha_bar_t) / sqrt(alpha_bar_t)
        self.x_0_pred_coef_2 = torch.sqrt(1.0 - self.alphas_cumprod) / torch.sqrt(self.alphas_cumprod)

        # Posterior mean coefficients for q(x_{t-1} | x_t, x_0)
        # coef1 = sqrt(alpha_bar_{t-1}) * beta_t / (1 - alpha_bar_t)
        self.posterior_mean_coef1 = (
            torch.sqrt(self.alphas_cumprod_prev) * self.betas / (1.0 - self.alphas_cumprod)
        )
        # coef2 = sqrt(alpha_t) * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        self.posterior_mean_coef2 = (
            torch.sqrt(alphas) * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        # Posterior variance: beta_tilde_t = (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * beta_t
        self.posterior_variance = (
            (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod) * self.betas
        )
        self.posterior_log_variance_clipped = torch.log(
            self.posterior_variance.clamp(min=1e-20)
        )

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

    def get_posterior_parameters(self, x_0, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x_t, t):
        # Predict noise using denoising network
        pred_noise = self.model(x_t, t)
        # Predict x_0 from x_t and pred_noise
        x_0 = (
            extract(self.x_0_pred_coef_1, t, x_t.shape) * x_t
            - extract(self.x_0_pred_coef_2, t, x_t.shape) * pred_noise
        )
        x_0 = x_0.clamp(-1.0, 1.0)
        return (pred_noise, x_0)

    @torch.no_grad()
    def predict_denoised_at_prev_timestep(self, x, t: int):
        pred_noise, x_0 = self.model_predictions(x, t)
        posterior_mean, posterior_variance, posterior_log_variance_clipped = \
            self.get_posterior_parameters(x_0, x, t)
        # Sample x_{t-1}
        noise = torch.randn_like(x)
        # No noise at t=0
        nonzero_mask = (t != 0).float().reshape(-1, *((1,) * (len(x.shape) - 1)))
        pred_img = posterior_mean + nonzero_mask * torch.exp(0.5 * posterior_log_variance_clipped) * noise
        return pred_img, x_0

    @torch.no_grad()
    def sample_ddpm(self, shape, z):
        img = z
        for t in tqdm(range(self.num_timesteps - 1, 0, -1)):
            batched_times = torch.full((img.shape[0],), t, device=self.device, dtype=torch.long)
            img, _ = self.predict_denoised_at_prev_timestep(img, batched_times)
        img = unnormalize_to_zero_to_one(img)
        return img

    def sample_times(self, total_timesteps, sampling_timesteps):
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        return list(reversed(times.int().tolist()))

    def get_time_pairs(self, times):
        return list(zip(times[:-1], times[1:]))

    def ddim_step(self, batch, device, tau_i, tau_isub1, img, model_predictions, alphas_cumprod, eta):
        # Step 1: predict x_0 and noise at tau_i
        t = torch.full((batch,), tau_i, device=device, dtype=torch.long)
        pred_noise, x_0 = model_predictions(img, t)

        if tau_isub1 < 0:
            tau_isub1 = 0

        # Step 2: extract alpha_bar values
        alpha_bar_tau_i = alphas_cumprod[tau_i]
        alpha_bar_tau_isub1 = alphas_cumprod[tau_isub1]

        # Step 3: compute sigma_tau_i
        beta_tau_i = 1.0 - alpha_bar_tau_i / alpha_bar_tau_isub1
        sigma_tau_i = eta * torch.sqrt(
            (1.0 - alpha_bar_tau_isub1) / (1.0 - alpha_bar_tau_i) * beta_tau_i
        )

        # Step 4: coefficient of epsilon_tau_i (direction pointing to x_t)
        eps_coef = torch.sqrt(1.0 - alpha_bar_tau_isub1 - sigma_tau_i ** 2)

        # Step 5: sample x_{tau_{i-1}}
        mu = torch.sqrt(alpha_bar_tau_isub1) * x_0 + eps_coef * pred_noise
        z = torch.randn_like(img) if tau_isub1 > 0 else torch.zeros_like(img)
        img = mu + sigma_tau_i * z

        return img, x_0

    def sample_ddim(self, shape, z):
        batch, device = shape[0], self.device
        total_timesteps = self.num_timesteps
        sampling_timesteps = self.sampling_timesteps
        eta = self.ddim_sampling_eta

        times = self.sample_times(total_timesteps, sampling_timesteps)
        time_pairs = self.get_time_pairs(times)

        img = z
        for tau_i, tau_isub1 in tqdm(time_pairs, desc='sampling loop time step'):
            img, _ = self.ddim_step(
                batch, device, tau_i, tau_isub1, img,
                self.model_predictions, self.alphas_cumprod, eta
            )

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, shape):
        sample_fn = self.sample_ddpm if not self.is_ddim_sampling else self.sample_ddim
        z = torch.randn(shape, device=self.betas.device)
        return sample_fn(shape, z)

    @torch.no_grad()
    def sample_given_z(self, z, shape):
        sample_fn = self.sample_ddpm if not self.is_ddim_sampling else self.sample_ddim
        z = z.reshape(shape)
        return sample_fn(shape, z)
