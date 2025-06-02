import os
import numpy as np
import torch
from lib.model_gan import Discriminator, Generator


def test_gan():
    # define hyperparameters
    latent_size = 64
    batch_size = 2
    num_channels = 4

    # test generator
    latent_noise = torch.randn((batch_size, latent_size, 1, 1))
    generator = Generator(channels_multiplier=num_channels, latent_size=latent_size)
    generated_images = generator(latent_noise)
    true_shape = (batch_size, 3, 32, 32)
    assert (
        generated_images.shape == true_shape
    ), f"Generator output shape is {generated_images.shape} but should be {true_shape}"
    # count parameters
    g_params_truth = 43748
    g_params = np.sum([np.product(param.shape) for param in generator.parameters()])
    assert (
        g_params == g_params_truth
    ), f"Generator should have {g_params_truth} parameters but has {g_params}"

    # test discriminator
    images = torch.randn((batch_size, 3, 32, 32))
    discriminator = Discriminator(channels_multiplier=num_channels)
    output_disc = discriminator(images)
    true_shape_disc = (batch_size, 1, 1, 1)
    assert (
        output_disc.shape == true_shape_disc
    ), f"Discriminator output shape is {output_disc.shape} but should be {true_shape_disc}"
    # count parameters
    d_params_truth = 3056
    d_params = np.sum([np.product(param.shape) for param in discriminator.parameters()])
    assert (
        d_params == d_params_truth
    ), f"Discriminator should have {d_params_truth} parameters but has {d_params}"

    # testing outputs of Generator and Discriminator
    torch.manual_seed(1000)
    latent_noise = torch.randn((2, 4, 1, 1))
    images = torch.randn((2, 1, 32, 32))
    generator = Generator(channels_multiplier=1, latent_size=4, num_input_channels=1)
    discriminator = Discriminator(channels_multiplier=1, num_input_channels=1)
    generated_images = generator(latent_noise).detach().cpu().numpy()
    output_disc = discriminator(images).detach().cpu().numpy()
    true_output_disc = np.array([[[[0.33339936]]], [[[0.47757638]]]])
    true_generated_images = np.load(
        os.path.join(os.path.dirname(__file__), "generated_images.npy")
    )
    np.testing.assert_allclose(
        output_disc,
        true_output_disc,
        rtol=1e-06,
        err_msg="Output of Discriminator is incorrect",
    )
    np.testing.assert_allclose(
        generated_images,
        true_generated_images,
        rtol=1e-03,
        err_msg="Output of Generator is incorrect",
    )


if __name__ == "__main__":
    test_gan()
    print("Test complete.")
