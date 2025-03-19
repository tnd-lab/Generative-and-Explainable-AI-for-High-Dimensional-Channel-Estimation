import torch
from tqdm import tqdm

from src.data.get_data import topology_20000_samples
from src.data.loader import ChannelDataloader
from src.models.vae import VAE, OriginalVAE
from src.utils.mnse import nmse_from_vae
from src.utils.utils import compute_mean_std, load_config
from src.utils.plots import plot_nmse


if __name__ == "__main__":
    # load config
    config = load_config("src/configs/configs.yml")
    device = config["device"]
    z_dim = config["z_dim"]
    snr_values = config["snr_values"]

    all_nmse_scores = {}

    # get the data
    channels, responses, transmits = topology_20000_samples(
        directory=config["datasets"]
    )
    # load the data
    dataloader = ChannelDataloader(y_res=responses, channels_res=channels)
    test_dataset = dataloader.test_dataloader()
    train_dataset = dataloader.train_dataloader()

    ################
    # Original VAE #
    ################
    vae = OriginalVAE(z_dim=z_dim).to(device)
    checkpoint = torch.load(config["models"]["vae"]["model_path"], map_location=device)
    vae.load_state_dict(checkpoint["state_dict"])
    vae.eval()

    total_mnse_scores = []
    for snr in tqdm(snr_values):
        nmse_score = nmse_from_vae(vae, test_dataset, transmits, snr, device)
        total_mnse_scores.append(nmse_score)

    all_nmse_scores["VAE"] = total_mnse_scores

    #################
    # VAE + WGAN-GP #
    #################
    vae = OriginalVAE(z_dim=z_dim).to(device)
    checkpoint = torch.load(
        config["models"]["vae_wgan_gp"]["model_path"], map_location=device
    )
    vae.load_state_dict(checkpoint["state_dict"])
    vae.eval()

    total_mnse_scores = []
    for snr in tqdm(snr_values):
        nmse_score = nmse_from_vae(vae, test_dataset, transmits, snr, device)
        total_mnse_scores.append(nmse_score)

    all_nmse_scores["VAE + WGAN-GP"] = total_mnse_scores

    ################################
    # VAE + WGAN-GP: Skewness Loss #
    ################################
    vae = OriginalVAE(z_dim=z_dim).to(device)
    checkpoint = torch.load(
        config["models"]["vae_wgan_gp_skewness_loss"]["model_path"], map_location=device
    )
    vae.load_state_dict(checkpoint["state_dict"])
    vae.eval()

    total_mnse_scores = []
    for snr in tqdm(snr_values):
        nmse_score = nmse_from_vae(vae, test_dataset, transmits, snr, device)
        total_mnse_scores.append(nmse_score)

    all_nmse_scores["VAE + WGAN-GP: Skewness Loss"] = total_mnse_scores

    ################
    # VAE Proposed #
    ################
    # get mean and std of the channel on training set
    MU_H, STD_H = compute_mean_std(train_dataset.dataset.channels_res, axis=0)
    MU_H = torch.from_numpy(MU_H.reshape(-1)).float().to(device)
    STD_H = torch.from_numpy(STD_H.reshape(-1)).float().to(device)

    vae = VAE(z_dim=z_dim, mu_h=MU_H, std_h=STD_H).to(device)
    checkpoint = torch.load(
        config["models"]["vae_proposed"]["model_path"], map_location=device
    )
    vae.load_state_dict(checkpoint["state_dict"])
    vae.eval()

    total_mnse_scores = []
    for snr in tqdm(snr_values):
        nmse_score = nmse_from_vae(vae, test_dataset, transmits, snr, device)
        total_mnse_scores.append(nmse_score)

    all_nmse_scores["VAE Proposed"] = total_mnse_scores

    ##########################
    # VAE Proposed + WGAN-GP #
    ##########################
    vae = VAE(z_dim=z_dim, mu_h=MU_H, std_h=STD_H).to(device)
    checkpoint = torch.load(
        config["models"]["vae_proposed_wgan_gp"]["model_path"], map_location=device
    )
    vae.load_state_dict(checkpoint["state_dict"])
    vae.eval()

    total_mnse_scores = []
    for snr in tqdm(snr_values):
        nmse_score = nmse_from_vae(vae, test_dataset, transmits, snr, device)
        total_mnse_scores.append(nmse_score)

    all_nmse_scores["VAE Proposed + WGAN-GP"] = total_mnse_scores

    ################################
    # VAE + WGAN-GP: Skewness Loss #
    ################################
    vae = VAE(z_dim=z_dim, mu_h=MU_H, std_h=STD_H).to(device)
    checkpoint = torch.load(
        config["models"]["vae_proposed_wgan_gp_skewness_loss"]["model_path"],
        map_location=device,
    )
    vae.load_state_dict(checkpoint["state_dict"])
    vae.eval()

    total_mnse_scores = []
    for snr in tqdm(snr_values):
        nmse_score = nmse_from_vae(vae, test_dataset, transmits, snr, device)
        total_mnse_scores.append(nmse_score)

    all_nmse_scores["VAE Proposed + WGAN-GP: Skewness Loss"] = total_mnse_scores

    plot_nmse(all_nmse_scores, snr_values, "./results/test/nmse_vae_wgan_gp.png")
