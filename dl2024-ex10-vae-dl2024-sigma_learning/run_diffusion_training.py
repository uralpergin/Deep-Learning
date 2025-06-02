import argparse

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch

from lib.model_diffusion import SimpleDiffusion
from lib.train_diffusion import train
from lib.utils import cosine_schedule, get_dataset, linear_schedule, sample_ddpm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def draw_frame(i):
    plt.clf()
    Xvis = Xgen_hist[i].cpu()
    fig = plt.scatter(
        Xvis[:, 0], Xvis[:, 1], marker="1", animated=True, label=f"frame={i}"
    )
    plt.legend(loc="upper right")
    return (fig,)


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
                1
                / (alphas[t] ** 0.5)
                * (x - (1 - alphas[t]) / ((1 - baralphas[t]) ** 0.5) * predicted_noise)
            )
            if t > 1:
                # See DDPM paper section 3.2.
                # Choosing the variance through beta_t is optimal for x_0 a normal distribution
                variance = betas[t]
                std = variance ** (0.5)
                x += std * torch.randn(size=(n_samples, n_features)).to(device)
            xt.append(x)
        return x, xt


def main():
    parser = argparse.ArgumentParser("simple 2d diffusion")
    parser.add_argument(
        "--dataset",
        type=str,
        default="moons",
        help='Which dataset to use. Choose from ["moons", "swiss_roll", "circles"]. '
        'Default is "moons".',
    )
    parser.add_argument(
        "--batch_size", type=int, default=2048, help="Batch size. Default is 2048."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Initial learning rate"
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="cosine",
        help="Schedule to use for beta cosine or linear. Default is cosine.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=250,
        help="Number of training epochs. Default is 250.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100000,
        help="Number of samples in dataset. Default is 100000.",
    )
    parser.add_argument(
        "--n_blocks",
        type=int,
        default=8,
        help="Number of linear layers in diffusion model. Default is 8.",
    )
    parser.add_argument(
        "--diffusion_steps",
        type=int,
        default=100,
        help="Number of diffusion steps. Default is 100.",
    )
    parser.add_argument(
        "--no_anim",
        action="store_true",
        help="Disable animation to speed up execution. Can be used for testing purposes. Default is False.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level, set to 0 to disable. Default is 1.",
    )
    args = parser.parse_args()

    print("=" * 80)
    print(
        f"Dataset: {args.dataset}"
        f"\nEpochs: {args.epochs}"
        f"\nSchedule for beta: {args.schedule}"
        f"\nDiffusion steps: {args.diffusion_steps}"
        f"\nNumber of samples in the dataset: {args.n_samples}"
    )
    print("=" * 80)

    dataset = get_dataset(args.dataset, args.n_samples)
    dataset = torch.tensor(dataset, dtype=torch.float32).to(device)

    if args.schedule == "cosine":
        baralphas, alphas, betas = cosine_schedule(args.diffusion_steps)
    else:
        baralphas, alphas, betas = linear_schedule(args.diffusion_steps)

    model = SimpleDiffusion(n_features=2, n_blocks=args.n_blocks)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train(
        args.dataset,  # dataset_name
        dataset,
        model,
        optimizer,
        baralphas,
        args.batch_size,
        args.epochs,
        args.diffusion_steps,
        alphas,
        betas,
        args.verbose,
    )
    global Xgen_hist
    Xgen, Xgen_hist = sample_ddpm(
        args.diffusion_steps, model, 100000, 2, baralphas, alphas, betas
    )
    Xgen = Xgen.cpu()

    plt.scatter(
        Xgen[:, 0],
        Xgen[:, 1],
        alpha=0.4,
        edgecolors="none",
        s=10,
        label="Generated data",
    )
    plt.scatter(
        dataset[:, 0].cpu(),
        dataset[:, 1].cpu(),
        alpha=0.4,
        edgecolors="none",
        s=10,
        label="Original data",
    )
    plt.legend()
    plt.savefig(f"{args.dataset}_generated_samples.png")
    plt.clf()

    if not args.no_anim:
        fig = plt.figure()
        anim = animation.FuncAnimation(
            fig, draw_frame, frames=args.diffusion_steps, interval=20, blit=True
        )
        anim_name = "{}_generation.mp4".format(args.dataset)
        print("Saving animation to {}".format(anim_name))
        anim.save(anim_name, fps=10)


if __name__ == "__main__":
    main()
