from typing import Tuple

import einops
import torch
from torch import Tensor, nn
from torch.distributions import Normal
from torch.nn.functional import mse_loss
from tqdm.auto import tqdm

from . import noise_utils


def linear_schedule(timesteps: int, start: float = 0.0001, end: float = 0.02) -> Tensor:
    r"""constants increasing linearly from :math:`10^{-4}` to :math:`0.02`

    Args:
        timesteps: total timesteps
        start: starting value, defaults to 0.0001
        end: end value, defaults to 0.02

    Returns:
        a 1d tensor representing :math:`\beta_t` indexed by :math:`t`
    """
    beta = torch.linspace(start, end, timesteps)
    return noise_utils.pad(beta)


def forward_process(image: Tensor, alpha_bar_t: Tensor) -> Normal:
    r"""Forward Process, :math:`q(x_t|x_{t-1})`

    Args:
        image: image of shape :math:`(N, C, H, W)`
        alpha_bar_t: :math:`\bar\alpha_t` of shape :math:`(N, 1, 1, *)`
        noise: noise sampled from standard normal distribution with the same shape as the image

    Returns:
        gaussian transition distirbution :math:`q(x_t|x_{t-1})`
    """

    mean = torch.sqrt(alpha_bar_t) * image

    variance = 1 - alpha_bar_t
    std = torch.sqrt(variance)

    return Normal(mean, std)


def reverse_process(
    x_t: Tensor,
    beta_t: Tensor,
    alpha_t: Tensor,
    alpha_bar_t: Tensor,
    noise_in_x_t: Tensor,
    variance: Tensor,
) -> Normal:
    r"""Reverse Denoising Process, :math:`p_\theta(x_{t-1}|x_t)`

    Args:
        beta_t: :math:`\beta_t` of shape :math:`(N, 1, 1, *)`
        alpha_t: :math:`\alpha_t` of shape :math:`(N, 1, 1, *)`
        alpha_bar_t: :math:`\bar\alpha_t` of shape :math:`(N, 1, 1, *)`
        noise_in_x_t: estimated noise in :math:`x_t` predicted by a neural network
        variance: variance of the reverse process, either learned or fixed
        noise: noise sampled from :math:`\mathcal{N}(0, I)`

    Returns:
        denoising distirbution :math:`q(x_t|x_{t-1})`
    """

    mean = (
        1
        / torch.sqrt(alpha_t)
        * (x_t - beta_t / torch.sqrt(1 - alpha_bar_t) * noise_in_x_t)
    )
    std = torch.sqrt(variance)
    return Normal(mean, std)


def simple_loss(noise: Tensor, estimated_noise: Tensor) -> Tensor:
    r"""Simple Loss objective :math:`L_\text{simple}`, MSE loss between noise and predicted noise

    Args:
        noise (torch.Tensor): noise used in the forward process
        estimated_noise (torch.Tensor): estimated noise with the same shape as :code:`noise`

    """
    return mse_loss(noise, estimated_noise)


class DDPM(nn.Module):
    r"""Training and Sampling for DDPM

    Args:
        model: model predicting noise from data, :math:`\epsilon_\theta(x_t, t)`
        timesteps: total timesteps :math:`T`
        start: linear variance schedule start value
        end: linear variance schedule end value
    """

    beta: Tensor
    alpha: Tensor
    alpha_bar: Tensor

    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        start: float = 0.0001,
        end: float = 0.02,
    ) -> None:
        super().__init__()

        self.model = model
        self.timesteps = timesteps

        beta = linear_schedule(timesteps, start, end)
        beta = einops.rearrange(beta, "t -> t 1 1 1")

        alpha = 1 - beta

        # alpha[0] = 1 so no problems here
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffer("beta", beta, persistent=False)
        self.register_buffer("alpha", alpha, persistent=False)
        self.register_buffer("alpha_bar", alpha_bar, persistent=False)

    def training_step(self, x_0: Tensor, y_0: Tensor) -> Tensor:
        r"""Training step except for optimization

        Args:
            x_0: motion spectrogram
            y_0: mel spectrogram

        Returns:
            loss, :math:`L_\text{simple}`
        """

        batch_size = x_0.size(0)

        time = noise_utils.uniform_int(
            1,
            self.timesteps,
            batch_size,
            device=x_0.device,
        )

        alpha_bar_t = self.alpha_bar[time]

        q_x = forward_process(x_0, alpha_bar_t)
        x_t = q_x.sample()

        q_y = forward_process(y_0, alpha_bar_t)
        y_t = q_y.sample()

        noise_in_x_t, noise_in_y_t = self.model(x_t, y_t, time)

        x_noise = (x_t - q_x.mean) / q_x.stddev
        y_noise = (y_t - q_y.mean) / q_y.stddev

        x_loss = simple_loss(x_noise, noise_in_x_t)
        y_loss = simple_loss(y_noise, noise_in_y_t)

        total_loss = x_loss + y_loss * 0.01
        return total_loss, {"x_loss": x_loss, "y_loss": y_loss}

    def sampling_step(self, x_t: Tensor, t: Tensor) -> Tensor:
        r"""Denoise image by sampling from :math:`p_\theta(x_{t-1}|x_t)`

        Args:
            x_t: image of shape :math:`(N, C, H, W)`
            t: starting :math:`t` to sample from, a tensor of shape :math:`(N,)`

        Returns:
            denoised image of shape :math:`(N, C, H, W)`
        """

        beta_t = self.beta[t]
        alpha_t = self.alpha[t]
        alpha_bar_t = self.alpha_bar[t]

        noise_in_x_t = self.model(x_t, t)
        p = reverse_process(
            x_t,
            beta_t,
            alpha_t,
            alpha_bar_t,
            noise_in_x_t,
            variance=beta_t,
        )
        x_t = p.sample()

        # set z to 0 when t = 1 by overwriting values
        x_t = torch.where(t == 1, p.mean, x_t)
        return x_t

    def generate(self, img_size: Tuple[int, int, int, int]) -> Tensor:
        """Generate image of shape :math:`(N, C, H, W)` by running the full denoising steps

        Args:
            img_size: image size to generate as a tuple :math:`(N, C, H, W)`

        Returns:
            generated image of shape :math:`(N, C, H, W)` as a tensor
        """

        x_t = noise_utils.gaussian(img_size, device=self.beta.device)
        all_t = torch.arange(
            0,
            self.timesteps + 1,
            device=self.beta.device,
        ).unsqueeze(dim=1)

        for t in tqdm(range(self.timesteps, 0, -1), leave=False):
            x_t = self.sampling_step(x_t, all_t[t])

        return x_t

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Applies forward to internal model

        Args:
            x: input image passed to internal model
            t: timestep passed to internal model
        """

        noise_in_x = self.model(x, t)
        return noise_in_x
