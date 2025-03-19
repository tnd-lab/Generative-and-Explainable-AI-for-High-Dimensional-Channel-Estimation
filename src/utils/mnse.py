import numpy as np
import tensorflow as tf
import torch
from tqdm import tqdm

from src.data.get_data import generate_responses


def nmse_func(pred, target):
    """
    Compute the Normalized Mean Squared Error (NMSE) between two tensors.

    Args:
        pred (torch.Tensor): Predicted tensor.
        target (torch.Tensor): Ground truth tensor (target).

    Returns:
        float: The NMSE value.
    """
    # Compute the Mean Squared Error (MSE)
    mse = torch.mean((pred - target) ** 2)

    # Compute the norm of the target tensor
    norm = torch.mean(target**2)

    # NMSE is the ratio of MSE to the norm of the target
    nmse_value = mse / norm

    return nmse_value.item()


def nmse_from_vae(model, datasets, transmits, snr, device):
    current_channels = datasets.dataset.channels_res
    current_responses = generate_responses(
        current_channels,
        transmits[: current_channels.shape[0]],
        current_channels.shape[0],
        snr=snr,
    )

    datasets.dataset.y_res = current_responses

    nmse_scores = []
    for i, (channel, response) in enumerate(datasets):
        channel = channel.to(device)
        response = response.to(device)

        fake_channel, *values = model(response)
        nmse_score = nmse_func(
            pred=fake_channel.detach().cpu(),
            target=channel.detach().cpu(),
        )
        nmse_scores.append(nmse_score)

    return np.mean(nmse_scores)


def nmse_from_gan(model, datasets, transmits, snr, device):
    current_channels = datasets.dataset.channels_res
    current_responses = generate_responses(
        current_channels,
        transmits[: current_channels.shape[0]],
        current_channels.shape[0],
        snr=snr,
    )

    datasets.dataset.y_res = current_responses

    nmse_scores = []
    for i, (channel, response) in enumerate(datasets):
        channel = channel.to(device)
        response = response.to(device)

        fake_channel = model(response)
        nmse_score = nmse_func(
            pred=fake_channel.detach().cpu(),
            target=channel.detach().cpu(),
        )
        nmse_scores.append(nmse_score)

    return np.mean(nmse_scores)


def evaluate_mse(model, snr_dbs, batch_size, num_it):

    # Casting model inputs to TensorFlow types to avoid
    # re-building of the graph
    snr_dbs = tf.cast(snr_dbs, tf.float32)
    batch_size = tf.cast(batch_size, tf.int32)

    mses = []
    for snr_db in tqdm(snr_dbs):

        mse_ = 0.0

        for _ in range(num_it):
            mse_ += model(batch_size, snr_db).numpy()
        # Averaging over the number of iterations
        mse_ /= float(num_it)
        mses.append(mse_)

    return mses
