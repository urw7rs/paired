import jax
import jax.numpy as jnp
from einops import rearrange
from flax import struct


Array = jax.Array


@struct.dataclass
class Linear:
    r"""constants increasing linearly from :math:`10^{-4}` to :math:`0.02`

    Args:
        timesteps: total timesteps
        start: starting value, defaults to 0.0001
        end: end value, defaults to 0.02

    Returns:
        a 1d tensor representing :math:`\beta_t` indexed by :math:`t`
    """
    beta: Array
    alpha: Array
    alpha_bar: Array
    timesteps: int

    @classmethod
    def create(
        cls, timesteps: int, start: float = 0.0001, end: float = 0.02, dtype=None
    ):
        beta = jnp.linspace(start, end, num=timesteps, dtype=dtype)
        beta = jnp.pad(beta, pad_width=(1, 0))
        beta = rearrange(beta, "t -> t 1 1 1")
        alpha = 1 - beta
        alpha_bar = jnp.cumprod(alpha, axis=0)
        timesteps = beta.shape[0] - 1
        return cls(beta, alpha, alpha_bar, timesteps)


def forward_process(alpha_bar_t, x, noise):
    r"""Forward Process, :math:`q(x_t|x_{t-1})`

    Args:
        x: image of shape :math:`(N, C, H, W)`
        noise: noise sampled from standard normal distribution with the same shape as the image
        alpha_bar_t: :math:`\bar\alpha_t` of shape :math:`(N, 1, 1, *)`

    Returns:
        gaussian transition distirbution :math:`q(x_t|x_{t-1})`
    """
    mean = jnp.sqrt(alpha_bar_t) * x
    stddev = jnp.sqrt(1 - alpha_bar_t)
    return mean + stddev * noise


def reverse_process(x_t, noise, beta_t, alpha_t, alpha_bar_t, noise_in_x_t):
    r"""Reverse Denoising Process, :math:`p_\theta(x_{t-1}|x_t)`

    Args:
        x_t: :math:`\x_t` of shape :math:`(N, H, W, C)`
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
        / jnp.sqrt(alpha_t)
        * (x_t - beta_t / jnp.sqrt(1 - alpha_bar_t) * noise_in_x_t)
    )
    stddev = jnp.sqrt(beta_t)
    return mean + stddev * noise
