import os
import pickle

import gdown
import numpy as np
import tensorflow as tf
import torch
import yaml


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_topology(filename):
    with open("topology.pkl", "rb") as f:
        loaded_topology = pickle.load(f)

    # Convert all numpy arrays to TensorFlow tensors
    for key, value in loaded_topology.items():
        loaded_topology[key] = tf.convert_to_tensor(value)

    return loaded_topology


def download_and_load_pkl(url, output="output.pkl"):
    # Download the pickle file from Google Drive

    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

    # Load the pickle file
    with open(output, "rb") as f:
        data = pickle.load(f)

    return data


def current_noise(signal, snr):
    snr = 10 ** (snr / 10)

    # signal_power = np.abs(np.mean(signal**2))
    signal_power = 0.01

    noise_power = signal_power / snr

    noise = (
        np.sqrt(noise_power)
        * (
            np.random.normal(size=signal.shape)
            + 1j * np.random.normal(size=signal.shape)
        )
        / np.sqrt(2)
    )

    return noise


def compute_mean_std(datasets, axis=0):
    mu_datasets = datasets.mean(axis=axis)
    std_datasets = datasets.std(axis=axis)

    return mu_datasets, std_datasets


def save_checkpoint(model, file_name):
    model_to_save = {
        "state_dict": model.state_dict(),
    }
    if not os.path.exists(file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
    torch.save(model_to_save, file_name)


def save_npy(file_name, np_array):
    if not os.path.exists(file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
    np.save(file_name, np_array)
