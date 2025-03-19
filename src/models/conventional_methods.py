import pickle

import gdown
import numpy as np
import sionna
import tensorflow as tf
import torch
from sionna.channel import GenerateOFDMChannel, OFDMChannel, gen_single_sector_topology
from sionna.channel.tr38901 import Antenna, PanelArray, UMi
from sionna.fec.ldpc import LDPC5GEncoder
from sionna.mapping import Mapper
from sionna.mimo import StreamManagement
from sionna.ofdm import (
    LMMSEInterpolator,
    LSChannelEstimator,
    ResourceGrid,
    ResourceGridMapper,
)
from sionna.utils import BinarySource, QAMSource, compute_ser, ebnodb2no, sim_ber
from tensorflow.keras import Model
from tqdm import tqdm

from src.utils.utils import download_and_load_pkl, load_config, load_topology


def get_resource_grid(
    num_ofdm_symbols: int = 14,
    fft_size: int = 612,
    num_tx: int = 1,
    cyclic_prefix_length: int = 0,
    subcarrier_spacing: int = 30e3,
    pilot_pattern: int = "kronecker",
    pilot_ofdm_symbol_indices: list = [2, 11],
):
    rg = ResourceGrid(
        num_ofdm_symbols=num_ofdm_symbols,
        fft_size=fft_size,
        num_tx=num_tx,
        cyclic_prefix_length=cyclic_prefix_length,
        subcarrier_spacing=subcarrier_spacing,
        pilot_pattern=pilot_pattern,
        pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices,
    )
    return rg


@tf.function(jit_compile=True)  # Use XLA for speed-up
def estimate_covariance_matrices(
    current_channels, number_antennas, num_ofdm_symbols, fft_size, num_it
):
    h_freq = np.zeros(
        (current_channels.shape[0], number_antennas, num_ofdm_symbols, fft_size),
        dtype=np.complex64,
    )
    h_freq.real = current_channels[..., 0]
    h_freq.imag = current_channels[..., 1]

    freq_cov_mat = tf.zeros([fft_size, fft_size], tf.complex64)
    time_cov_mat = tf.zeros([num_ofdm_symbols, num_ofdm_symbols], tf.complex64)
    space_cov_mat = tf.zeros([number_antennas, number_antennas], tf.complex64)
    for _ in tf.range(num_it):
        # [batch size, num_rx_ant, num_ofdm_symbols, fft_size]
        h_samples = h_freq.reshape(
            current_channels.shape[0], number_antennas, num_ofdm_symbols, fft_size
        )
        #################################
        # Estimate frequency covariance
        #################################
        # [batch size, num_rx_ant, fft_size, num_ofdm_symbols]
        h_samples_ = tf.transpose(h_samples, [0, 1, 3, 2])
        # [batch size, num_rx_ant, fft_size, fft_size]
        freq_cov_mat_ = tf.matmul(h_samples_, h_samples_, adjoint_b=True)
        # [fft_size, fft_size]
        freq_cov_mat_ = tf.reduce_mean(freq_cov_mat_, axis=(0, 1))
        # [fft_size, fft_size]
        freq_cov_mat += freq_cov_mat_

        ################################
        # Estimate time covariance
        ################################
        # [batch size, num_rx_ant, num_ofdm_symbols, fft_size]
        time_cov_mat_ = tf.matmul(h_samples, h_samples, adjoint_b=True)
        # [num_ofdm_symbols, num_ofdm_symbols]
        time_cov_mat_ = tf.reduce_mean(time_cov_mat_, axis=(0, 1))
        # [num_ofdm_symbols, num_ofdm_symbols]
        time_cov_mat += time_cov_mat_

        ###############################
        # Estimate spatial covariance
        ###############################
        # [batch size, num_ofdm_symbols, num_rx_ant, fft_size]
        h_samples_ = tf.transpose(h_samples, [0, 2, 1, 3])
        # [batch size, num_ofdm_symbols, num_rx_ant, num_rx_ant]
        space_cov_mat_ = tf.matmul(h_samples_, h_samples_, adjoint_b=True)
        # [num_rx_ant, num_rx_ant]
        space_cov_mat_ = tf.reduce_mean(space_cov_mat_, axis=(0, 1))
        # [num_rx_ant, num_rx_ant]
        space_cov_mat += space_cov_mat_

    freq_cov_mat /= tf.complex(tf.cast(fft_size * num_it, tf.float32), 0.0)
    time_cov_mat /= tf.complex(tf.cast(num_ofdm_symbols * num_it, tf.float32), 0.0)
    space_cov_mat /= tf.complex(tf.cast(number_antennas * num_it, tf.float32), 0.0)

    return freq_cov_mat, time_cov_mat, space_cov_mat


def ls_channel(response, transmit, epsilon=1e-10):
    mask = np.abs(transmit) <= epsilon

    estimated_channel = np.zeros_like(response, dtype=complex)

    estimated_channel[~mask] = response[~mask] / transmit[~mask]
    return estimated_channel


def ls_method(current_responses, current_transmits, number_of_sample):
    complex_responses = np.zeros(
        (number_of_sample, *current_responses.shape[1:-1])
    ).astype(np.complex64)
    complex_responses.real = current_responses[..., 0]
    complex_responses.imag = current_responses[..., 1]

    complex_transmits = np.zeros(
        (number_of_sample, *current_transmits.shape[1:-1])
    ).astype(np.complex64)
    complex_transmits.real = current_transmits[..., 0]
    complex_transmits.imag = current_transmits[..., 1]

    estimated_channels = np.zeros((number_of_sample, *current_transmits.shape[1:]))

    for i in range(number_of_sample):
        estimated_channel = ls_channel(complex_responses[i], complex_transmits[i])
        estimated_channels[i, ..., 0] = estimated_channel.real
        estimated_channels[i, ..., 1] = estimated_channel.imag

    return estimated_channels


class MIMOOFDMLink(Model):

    def __init__(self, int_method, lmmse_order=None, **kwargs):
        super().__init__(kwargs)

        assert int_method in ("nn", "lin", "lmmse")

        self._config = load_config("src/configs/configs.yml")
        self.topology = self._config["topology"]

        # The user terminals (UTs) are equipped with a single antenna
        # with vertial polarization.
        UT_ANTENNA = Antenna(
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",  # Omnidirectional antenna pattern
            carrier_frequency=self.topology["carrier_frequency"],
        )

        # The base station is equipped with an antenna
        # array of 8 cross-polarized antennas,
        # resulting in a total of 16 antenna elements.
        BS_ARRAY = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=4,
            polarization="single",
            polarization_type="V",
            antenna_pattern="38.901",  # 3GPP 38.901 antenna pattern
            carrier_frequency=self.topology["carrier_frequency"],
        )

        # 3GPP UMi channel model is considered
        self.CHANNEL_MODEL = UMi(
            carrier_frequency=self.topology["carrier_frequency"],
            o2i_model="low",
            ut_array=UT_ANTENNA,
            bs_array=BS_ARRAY,
            direction="uplink",
            enable_shadow_fading=False,
            enable_pathloss=True,
        )

        # sionna.Config.xla_compat = True # Enable Sionna's support of XLA
        FREQ_COV_MAT, TIME_COV_MAT, SPACE_COV_MAT = estimate_covariance_matrices(100)
        # sionna.Config.xla_compat = False # Disable Sionna's support of XLA

        # Configure the resource grid
        rg = ResourceGrid(
            num_ofdm_symbols=self.topology["num_ofdm_symbols"],
            fft_size=self.topology["fft_size"],
            subcarrier_spacing=self.topology["subcarrier_spacing"],
            num_tx=1,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[2, 6],
        )
        self.rg = rg

        # Stream management
        # Only a sinlge UT is considered for channel estimation
        # sm = StreamManagement([[1]], 1)

        ##################################
        # Transmitter
        ##################################

        self.qam_source = QAMSource(
            num_bits_per_symbol=2
        )  # Modulation order does not impact the channel estimation. Set to QPSK
        self.rg_mapper = ResourceGridMapper(rg)

        self.binary_source = BinarySource()
        self.mapper = Mapper(constellation_type="qam", num_bits_per_symbol=2)

        ##################################
        # Channel
        ##################################

        self.channel = OFDMChannel(self.CHANNEL_MODEL, rg, return_channel=True)

        # Channel estimation methods
        freq_cov_mat = tf.constant(FREQ_COV_MAT, tf.complex64)
        time_cov_mat = tf.constant(TIME_COV_MAT, tf.complex64)
        space_cov_mat = tf.constant(SPACE_COV_MAT, tf.complex64)
        if int_method == "nn":
            self.channel_estimator = LSChannelEstimator(rg, interpolation_type="nn")
        elif int_method == "lin":
            self.channel_estimator = LSChannelEstimator(rg, interpolation_type="lin")
        elif int_method == "lmmse":
            lmmse_int_freq_first = LMMSEInterpolator(
                rg.pilot_pattern,
                time_cov_mat,
                freq_cov_mat,
                space_cov_mat,
                order=lmmse_order,
            )
            self.channel_estimator = LSChannelEstimator(
                rg, interpolator=lmmse_int_freq_first
            )

    @tf.function
    def call(self, batch_size, snr_db, y_rg=None):

        ##################################
        # Transmitter
        ##################################

        x = self.qam_source([batch_size, 1, 1, self.rg.num_data_symbols])
        x_rg = self.rg_mapper(x)

        ##################################
        # Channel
        ##################################

        no = tf.pow(10.0, -snr_db / 10.0)
        topology = download_and_load_pkl(
            self.topology["url"],
            output="topology.pkl",
        )
        values_topology = list(load_topology(topology).values())
        self.CHANNEL_MODEL.set_topology(*values_topology)

        y_rg, h_freq = self.channel((x_rg, no))
        ###################################
        # Channel estimation
        ###################################

        h_hat, _ = self.channel_estimator((y_rg, no))

        ###################################
        # MSE
        ###################################

        mse = tf.reduce_mean(tf.square(tf.abs(h_freq - h_hat)))

        return mse
