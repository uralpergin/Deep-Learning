import numpy as np
import torch

from lib.model_diffusion import SimpleDiffusion
from lib.train_diffusion import train


def test_train():
    # Seed the environment
    np.random.seed(0)
    torch.manual_seed(0)
    # test case setup
    dataset_name = "test"
    batch_size = 2
    epochs = 5
    diffusion_steps = 10
    n_samples = 32
    n_blocks = 2

    # sample random dataset
    dataset = torch.randn(n_samples, 2)
    # calculate baralphas, alphas, betas
    betas = torch.linspace(0.0001, 0.02, diffusion_steps)
    alphas = 1 - betas
    baralphas = torch.cumprod(alphas, dim=0)
    # initialize dummy model
    model = SimpleDiffusion(n_features=2, n_blocks=n_blocks)
    # initialize dummy optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = train(
        dataset_name,
        dataset,
        model,
        optimizer,
        baralphas,
        batch_size,
        epochs,
        diffusion_steps,
        alphas,
        betas,
    )
    assert np.isclose(
        loss.item(), 1.825563669204712, rtol=1.0e-2
    ), f"Current loss does not match the hardcoded loss value, please recheck the implementation"


if __name__ == '__main__':
    test_train()
    print("Test complete.")
