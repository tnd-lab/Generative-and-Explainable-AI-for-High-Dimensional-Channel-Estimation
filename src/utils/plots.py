import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def show_tensor_images(image_tensor):
    """
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    """
    image_grid = image_tensor.detach().cpu()
    plt.imshow(image_grid.squeeze())
    # plt.show()
    plt.close()


def show_loss_images(c_losses=None, g_losses=None):
    plt.figure(figsize=(10, 5))
    if c_losses:
        plt.plot(c_losses, label="Discriminator Loss", linestyle="-", linewidth=2)
    if g_losses:
        plt.plot(g_losses, label="Generator Loss", linestyle="-", linewidth=2)
    plt.title("GAN Training Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.close()


def show_nmse(nmse_scores):
    plt.figure(figsize=(10, 5))
    plt.plot(
        range(len(nmse_scores)),
        nmse_scores,
        marker="o",
        linestyle="-",
        color="b",
        label="NMSE",
    )
    plt.title("Normalized Mean Squared Error (NMSE) vs Prediction Set", fontsize=14)
    plt.xlabel("Prediction Set", fontsize=12)
    plt.ylabel("NMSE", fontsize=12)
    plt.grid(True)
    plt.legend()
    # plt.show()
    plt.close()


def plot_generated_image(prediction, test_input, tar, t=0, prefix: str = "wgan"):

    test_input = test_input.cpu().detach().numpy().squeeze()
    prediction = prediction.cpu().detach().numpy().squeeze()
    tar = tar.cpu().detach().numpy().squeeze()

    display_list = [
        np.squeeze(test_input),
        np.squeeze(tar),
        np.squeeze(prediction),
    ]
    title = ["Input Y", "Target H", "Prediction H"]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis("off")

    img_path = os.path.join("results/generated_img", prefix, "img_" + str(t) + ".png")

    if not os.path.exists(img_path):
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
    plt.savefig(img_path)
    plt.close()


def plot_skewness_func(skewness_scores, epochs, img_path):
    # # Plot the loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, skewness_scores, label="Target")
    plt.xlabel("Epoch")
    plt.ylabel("Skewness distances")
    plt.title("Skewness Loss")
    plt.legend()
    if not os.path.exists(img_path):
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
    plt.savefig(img_path)
    plt.close()
    # plt.show()


def plot_nmse_func(mse_scores, epochs, img_path):
    # # Plot the loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mse_scores, label="Target")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("MSE Scores")
    plt.legend()
    if not os.path.exists(img_path):
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
    plt.savefig(img_path)
    plt.close()
    # plt.show()


def plot_loss_func(losses_d=None, losses_g=None, epochs=None, img_path=None):
    # # Plot the loss curves
    plt.figure(figsize=(10, 6))
    if losses_d:
        plt.plot(epochs, losses_d, label="Discriminator Loss")
    if losses_g:
        plt.plot(epochs, losses_g, label="Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()

    if not os.path.exists(img_path):
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
    plt.savefig(img_path)
    plt.close()
    # plt.show()


def plot_distribution(
    c_real,
    c_fake,
    g_fake,
    c_real_threshold=None,
    c_fake_threshold=None,
    g_fake_threshold=None,
    save_path=None,
):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Histogram plot
    ax1.hist(c_real, bins=30, alpha=0.5, label="Real Critic")
    ax1.hist(c_fake, bins=30, alpha=0.5, label="Fake Critic")
    ax1.hist(g_fake, bins=30, alpha=0.5, label="Fake Generator")
    if c_real_threshold:
        ax1.axvline(
            c_real_threshold,
            color="r",
            linestyle="--",
            label="Threshold for Real Critic",
        )
    if c_fake_threshold:
        ax1.axvline(
            c_fake_threshold,
            color="g",
            linestyle="--",
            label="Threshold for Fake Critic",
        )
    if g_fake_threshold:
        ax1.axvline(
            g_fake_threshold,
            color="b",
            linestyle="--",
            label="Threshold for Fake Generator",
        )
    ax1.set_title("Histogram of Distributions with Thresholds")
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Frequency")
    ax1.legend()

    # KDE plot
    kde_c_real = gaussian_kde(c_real)
    kde_c_fake = gaussian_kde(c_fake)
    kde_g_fake = gaussian_kde(g_fake)

    x_range = np.linspace(
        min(c_real.min(), c_fake.min(), g_fake.min()),
        max(c_real.max(), c_fake.max(), g_fake.max()),
        200,
    )

    ax2.plot(x_range, kde_c_real(x_range), label="Real Critic")
    ax2.plot(x_range, kde_c_fake(x_range), label="Fake Critic")
    ax2.plot(x_range, kde_g_fake(x_range), label="Fake Generator")
    if c_real_threshold:
        ax1.axvline(
            c_real_threshold,
            color="r",
            linestyle="--",
            label="Threshold for Real Critic",
        )
    if c_fake_threshold:
        ax1.axvline(
            c_fake_threshold,
            color="g",
            linestyle="--",
            label="Threshold for Fake Critic",
        )
    if g_fake_threshold:
        ax1.axvline(
            g_fake_threshold,
            color="b",
            linestyle="--",
            label="Threshold for Fake Generator",
        )
    ax2.set_title("Kernel Density Estimation of Distributions with Thresholds")
    ax2.set_xlabel("Value")
    ax2.set_ylabel("Density")
    ax2.legend()

    plt.tight_layout()

    # save file
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    # plt.show()
    plt.close()


def plot_vae_in_out(input_tensor, output, z, encoder_output, true_output):
    # Convert to numpy arrays
    encoder_input_np = input_tensor.detach().cpu().numpy().flatten()
    encoder_output_np = encoder_output.detach().cpu().numpy().flatten()
    decoder_input_np = z.detach().cpu().numpy().flatten()
    decoder_output_np = output.detach().cpu().numpy().flatten()
    true_output_np = true_output.detach().cpu().numpy().flatten()

    # Create a figure with five subplots in a single row
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 4))
    fig.patch.set_facecolor("#2F2F2F")

    # Define the data and labels for each subplot
    data = [
        encoder_input_np,
        encoder_output_np,
        decoder_input_np,
        decoder_output_np,
        true_output_np,
    ]
    titles = [
        "Encoder Input",
        "Encoder Output",
        "Decoder Input (Latent Space)",
        "Decoder Output",
        "True Output",
    ]
    shapes = [
        input_tensor.shape,
        encoder_output.shape,
        z.shape,
        output.shape,
        true_output.shape,
    ]
    colors = ["cyan", "magenta", "yellow", "green", "red"]
    axes = [ax1, ax2, ax3, ax4, ax5]

    for ax, d, title, shape, color in zip(axes, data, titles, shapes, colors):
        kde = gaussian_kde(d)
        x_range = np.linspace(d.min(), d.max(), 200)

        ax.plot(x_range, kde(x_range), color=color)
        ax.fill_between(x_range, kde(x_range), alpha=0.5, color=color)

        ax.set_title(f"{title}\nShape: {shape}", color="white", fontsize=16)
        ax.set_xlabel("Value", color="white", fontsize=12)
        ax.set_ylabel("Density", color="white", fontsize=12)

        ax.grid(color="gray", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.tick_params(colors="white")

    plt.tight_layout()
    # plt.show()
    plt.close()


def plot_nmse(nmse_scores, snr_values, save_path=None):
    plt.figure(figsize=(20, 12))
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 25
    plt.rcParams["axes.labelsize"] = 25
    plt.rcParams["axes.titlesize"] = 25
    plt.rcParams["xtick.labelsize"] = 30
    plt.rcParams["ytick.labelsize"] = 30
    plt.rcParams["legend.fontsize"] = 25

    line_styles = [
        "-.o",
        "--o",
        "-o",
        "-s",
        "-x",
        "--x",
        "-v",
        "--v",
        "-.v",
        "--^",
        "-^",
        "-.^",
    ]
    colors = plt.cm.tab10(np.linspace(0, 1, len(nmse_scores)))

    # Plot all lines except the lowest one first
    for (model, nmse_values), line_style, color in zip(
        nmse_scores.items(), line_styles, colors
    ):
        if "VAE" in model:
            plt.semilogy(
                snr_values,
                nmse_values,
                line_style,
                label=model,
                color=color,
                linewidth=4,
                markersize=10,
            )
        else:
            plt.semilogy(
                snr_values,
                nmse_values,
                line_style,
                label=model,
                color=color,
                linewidth=2,
                markersize=10,
            )

    plt.xlabel("SNR (dB)")
    plt.ylabel("NMSE")
    plt.legend(loc="upper right")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlim(min(snr_values), max(snr_values))

    plt.tight_layout()

    # save file
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    # plt.show()
