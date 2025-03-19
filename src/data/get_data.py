import pickle

import numpy as np
import tensorflow as tf

from src.utils.utils import current_noise, download_and_load_pkl


def generate_responses(current_channels, current_transmits, number_of_sample, snr):
    complex_channels = np.zeros(
        (number_of_sample, *current_channels.shape[1:-1])
    ).astype(np.complex64)
    complex_channels.real = current_channels[..., 0]
    complex_channels.imag = current_channels[..., 1]

    complex_transmits = np.zeros(
        (number_of_sample, *current_transmits.shape[1:-1])
    ).astype(np.complex64)
    complex_transmits.real = current_transmits[..., 0]
    complex_transmits.imag = current_transmits[..., 1]

    complex_responses = np.zeros(
        (number_of_sample, *current_channels.shape[1:-1])
    ).astype(np.complex64)
    new_responses = np.zeros((number_of_sample, *current_channels.shape[1:]))

    for i in range(number_of_sample):
        complex_responses[i] = complex_channels[i] * complex_transmits[i]
        complex_responses[i] += current_noise(complex_responses[i], snr)

        new_responses[i, ..., 0] = complex_responses[i].real
        new_responses[i, ..., 1] = complex_responses[i].imag

    return new_responses


def topology_20000_samples(directory: str = "data/"):
    # channel sionna, pilot sionna 20000 samples with the same topology
    channels = download_and_load_pkl(
        "https://drive.google.com/uc?export=download&id=1OwPoI-pQjuNvDmxhGHkBAweRnR73zwcv",  # noqa
        output=f"{directory}/channels_res.pkl",
    )
    responses = download_and_load_pkl(
        "https://drive.google.com/uc?export=download&id=1lMpM8LJldgIDlXwqaPyg_YFte_bIs80x",  # noqa
        output=f"{directory}/y_res.pkl",
    )
    transmits = download_and_load_pkl(
        "https://drive.google.com/uc?export=download&id=17PN692_smNKzVTqGj0-Gdz-jatCsV9Un",  # noqa
        output=f"{directory}/rg_input.pkl",
    )
    transmits = np.tile(transmits[0], (20000, 1, 1, 1, 1))
    
    return channels, responses, transmits


if __name__ == "__main__":
    topology_20000_samples(directory="src/data/datasets/")
