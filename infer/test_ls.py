from src.models.conventional_methods import MIMOOFDMLink
from src.utils.mnse import evaluate_mse
from src.utils.utils import load_config

if __name__ == "__main__":
    # load config
    config = load_config("src/configs/configs.yml")
    snr_values = config["snr_values"]
    batch_size = config["batch_size"]

    ORDERS = [
        "s-t-f",  # Space - time - frequency
        "t-f-s",  # Time - frequency - space
        "t-f",  # Time - frequency (no spatial smoothing)
    ]

    MSES = {}
    all_nmse_scores = {}

    # Nearest-neighbor interpolation
    e2e = MIMOOFDMLink("nn")
    all_nmse_scores["LS + Nearest neighbour"] = evaluate_mse(
        e2e, snr_values, batch_size, 100
    )

    # Linear interpolation
    e2e = MIMOOFDMLink("lin")
    all_nmse_scores["LS + Linear"] = evaluate_mse(e2e, snr_values, batch_size, 100)

    # LMMSE
    for order in ORDERS:
        e2e = MIMOOFDMLink("lmmse", order)
        all_nmse_scores[f"LS + LMMSE: {order}"] = evaluate_mse(
            e2e, snr_values, batch_size, 100
        )

    print(all_nmse_scores)
