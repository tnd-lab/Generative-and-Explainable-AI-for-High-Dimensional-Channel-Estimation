import torch
from tqdm import tqdm

from src.data.get_data import topology_20000_samples
from src.data.loader import ChannelDataloaderGAN
from src.models.gan import Generator
from src.utils.mnse import nmse_from_gan
from src.utils.utils import load_config
from src.utils.plots import plot_nmse

if __name__ == "__main__":
    # load config
    config = load_config("src/configs/configs.yml")
    device = config["device"]
    z_dim = config["z_dim"]
    snr_values = config["snr_values"]

    # get the data
    channels, responses, transmits = topology_20000_samples(
        directory=config["datasets"]
    )

    ################
    # Orininal Gan #
    ################
    dataloader_gan = ChannelDataloaderGAN(y_res=responses, channels_res=channels)
    test_dataset_gan = dataloader_gan.test_dataloader()

    gan = Generator((32 * 32 * 2)).to(device)
    checkpoint = torch.load(
        config["models"]["wgan_gp"]["model_path"], map_location=device
    )
    gan.load_state_dict(checkpoint["state_dict"])
    gan.eval()

    total_mnse_scores = []
    for snr in tqdm(snr_values):
        nmse_score = nmse_from_gan(gan, test_dataset_gan, transmits, snr, device)
        total_mnse_scores.append(nmse_score)

    plot_nmse({"GAN": total_mnse_scores}, snr_values, "./results/test/nmse_wgan_gp.png")
