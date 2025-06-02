import torch

from lib.utils import add_noise, plot_bar_alpha, plot_dataset, sample_ddpm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    dataset_name: str,
    dataset: torch.Tensor,
    model: torch.nn.Module,
    optimizer: torch.optim,
    baralphas: torch.Tensor,
    batch_size: int,
    epochs: int,
    diffusion_steps: int,
    alphas: torch.Tensor,
    betas: torch.Tensor,
    verbose: int = 0,
) -> torch.Tensor:
    """
    Train the diffusion model.

    Args:
        dataset_name (str): Dataset name, used for plotting.
        dataset (torch.Tensor): Dataset to train on.
        model (torch.nn.Module): Diffusion model.
        optimizer (torch.optim): Optimizer.
        baralphas (torch.Tensor): Alpha bar noise parameter.
        batch_size (int): Batch size.
        epochs (int): Number of epochs to train for.
        diffusion_steps (int): Number of diffusion steps.
        verbose (int, optional): Verbosity level. Defaults to 0.

    Returns:
        torch.Tensor: Loss
    """
    plot_bar_alpha()

    if dataset_name != "test":
        plot_dataset(dataset.cpu(), dataset_name=dataset_name)
    model = model.to(device)
    loss_fn = torch.nn.MSELoss()
    baralphas = baralphas.to(device)
    loss = None

    # Sample at the beginning
    Xgen, _ = sample_ddpm(diffusion_steps, model, 100000, 2, baralphas, alphas, betas)
    Xgen = Xgen.cpu()
    plot_dataset(Xgen, dataset_name=f"{dataset_name}_0_generated")

    for epoch in range(1, epochs + 1):
        epoch_loss = steps = 0
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            # sample random timesteps for every sample
            timesteps = torch.randint(0, diffusion_steps, size=[len(batch), 1]).to(device)

            # START TODO #
            # 1. Compute the noisy samples at different timesteps using add_noise function
            # (Note you need to complete the add_noise function in utils.py)
            # 2. Use the defined diffusion model to predict the noise added at timestep t
            # 3. Compute the MSE between the predicted noise and the added noise (gausssian noise)
            noise, eps = add_noise(batch, timesteps, baralphas)

            pred_n = model(noise, timesteps)

            loss = loss_fn(pred_n, eps)
            # END TODO #
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            steps += 1
        if verbose > 0:
            print(f"Epoch {epoch} loss = {epoch_loss / steps}")

        # Sample every 50 epochs
        if epoch % 50 == 0:
            Xgen, _ = sample_ddpm(diffusion_steps, model, 100000, 2, baralphas, alphas, betas)
            Xgen = Xgen.cpu()
            plot_dataset(Xgen, dataset_name=f"{dataset_name}_{epoch}_generated")

    return loss
