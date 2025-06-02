import math

import matplotlib.pyplot as plt
import torch
from sklearn.datasets import make_circles, make_moons, make_swiss_roll

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_ddpm(
    diffusion_steps: int,
    model: torch.nn.Module,
    n_samples: int,
    n_features: int,
    baralphas: torch.Tensor,
    alphas: torch.Tensor,
    betas: torch.Tensor,
) -> tuple:
    """Sampler following the Denoising Diffusion Probabilistic Models method by Ho et al (Algorithm 2)

    Args:
        diffusion_steps (int): Diffusion steps.
        model (torch.nn.Module): Diffusion model.
        n_samples (int): Number of samples.
        n_features (int): Number of features.
        baralphas (torch.Tensor): Alpha bar noise parameter.
        alphas (torch.Tensor): Alpha noise parameter.
        betas (torch.Tensor): Noise variance.

    Returns:
        tuple: Generated samples and the history of all generated samples
    """
    baralphas = baralphas.to(device)
    alphas = alphas.to(device)
    betas = betas.to(device)
    with torch.no_grad():
        x = torch.randn(size=(n_samples, n_features)).to(device)
        xt = [x]
        for t in range(diffusion_steps - 1, 0, -1):
            predicted_noise = model(x, torch.full([n_samples, 1], t).to(device))
            # See DDPM paper between equations 11 and 12
            x = (
                1 / (alphas[t] ** 0.5)
                * (x - (1 - alphas[t]) / ((1 - baralphas[t]) ** 0.5) * predicted_noise)
            )
            if t > 1:
                # See DDPM paper section 3.2.
                # Choosing the variance through beta_t is optimal for x_0 a normal distribution
                variance = betas[t]
                std = variance ** (0.5)
                x += std * \
                    torch.randn(size=(n_samples, n_features)).to(device)
            xt.append(x)
        return x, xt


def get_circles(n_samples: int = 10000):
    x, _ = make_circles(noise=0.05, factor=0.5, random_state=1, n_samples=n_samples)
    return x


def get_moons(n_samples: int = 10000):
    x, _ = make_moons(n_samples=n_samples, noise=0.01)
    return x


def get_swiss_roll(n_samples: int = 10000):
    x, _ = make_swiss_roll(n_samples=n_samples, noise=0.5)
    x = x[:, [0, 2]]
    x = (x - x.mean()) / x.std()
    return x


def get_dataset(dataset_name: str = "moons", n_samples: int = 10000):
    if dataset_name == "moons":
        return get_moons(n_samples=n_samples)
    elif dataset_name == "swiss_roll":
        return get_swiss_roll(n_samples=n_samples)
    elif dataset_name == "circles":
        return get_circles(n_samples=n_samples)
    else:
        raise NotImplementedError


def plot_dataset(x, dataset_name: str = "moons"):
    plt.scatter(x[:, 0], x[:, 1], alpha=0.4, edgecolors='none', s=10)
    fig_name = dataset_name + ".png"
    plt.savefig(fig_name)
    plt.clf()


def cosine_schedule(diffusion_steps: int = 1000):
    steps = diffusion_steps + 1
    s = 0.008
    x = torch.linspace(0, diffusion_steps, steps)
    alphas_cumprod = torch.cos(((x / diffusion_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0.0001, 0.9999)
    alphas = 1 - betas
    baralphas = torch.cumprod(alphas, dim=0)
    return baralphas, alphas, betas


def linear_schedule(diffusion_steps: int = 1000):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(beta_start, beta_end, diffusion_steps)
    alphas = 1-betas
    baralphas = torch.cumprod(alphas, dim=0)
    return baralphas, alphas, betas


def plot_bar_alpha():
    cosine, _, _ = cosine_schedule()
    linear, _, _ = linear_schedule()
    plt.plot(cosine, color="blue", label="cosine")
    plt.plot(linear, color="orange", label="linear")
    plt.legend()
    plt.savefig("linear_and_cosine_schedules.png")
    plt.clf()


def add_noise(batch, t, baralphas):
    eps = torch.randn(size=batch.shape).to(batch.device)
    # START TODO ################
    # 1. Add noise to the batch using equation 4 from the DDPM paper https://arxiv.org/pdf/2006.11239.pdf
    # 2. Use the baralphas and the noise sampled in the previous step to compute the noised data
    noised = torch.sqrt(baralphas[t]) * batch + torch.sqrt(1 - baralphas[t]) * eps
    # END TODO ################
    return noised, eps


def plot_noised(batch, t, baralphas):
    noiselevel = t
    noised, eps = add_noise(batch, torch.full([len(batch), 1], fill_value=noiselevel), baralphas)
    plt.scatter(noised[:, 0], noised[:, 1], marker="*", alpha=0.5)
    plt.scatter(batch[:, 0], batch[:, 1], alpha=0.5)
    plt.legend(["Noised data", "Original data"])
    save_name = "true_vs_noisy_{}.png".format(str(t))
    plt.savefig(save_name)
    plt.clf()


def plot_denoised_vs_true(batch, noised, t, baralphas):
    eps = torch.randn(size=batch.shape)
    denoised = 1 / torch.sqrt(baralphas[t]) * (noised - torch.sqrt(1 - baralphas[t]) * eps)
    plt.scatter(batch[:, 0], batch[:, 1], alpha=0.5)
    plt.scatter(denoised[:, 0], denoised[:, 1], marker="1", alpha=0.5)
    plt.legend(["Original data", "Recovered original data"])
    plt.savefig("original_and_denoised.png")
    plt.clf()
