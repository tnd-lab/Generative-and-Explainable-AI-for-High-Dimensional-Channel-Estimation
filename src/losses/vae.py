# Assuming the VAE loss function is defined as follows:
import torch

from src.losses.gan import l2_loss


def vae_loss(recon_x, x, STD_H, MU_H, latent_vector, *args):
    recon_loss = l2_loss(recon_x, x)
    # KL divergence formula between two Gaussians
    var_l = latent_vector.var(axis=0)
    mu_l = latent_vector.mean(axis=0)
    var_h = STD_H**2
    mu_h = MU_H
    kl_div = 0.5 * torch.sum(
        torch.log(var_h / var_l) + (var_l + (mu_l - mu_h).pow(2)) / var_h - 1
    )
    return recon_loss + 0.1 * kl_div, recon_loss, kl_div
