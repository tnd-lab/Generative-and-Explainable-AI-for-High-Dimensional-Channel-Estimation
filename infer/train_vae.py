import numpy as np
import torch
import torch.nn as nn

from src.data.get_data import topology_20000_samples
from src.data.loader import ChannelDataloader
from src.losses.gan import gradient_penalty_loss, l2_loss
from src.losses.skewness import SkewCalculator
from src.losses.vae import vae_loss
from src.models import weights_init
from src.models.gan import Critic
from src.models.vae import VAE
from src.utils.explaination import GradCam
from src.utils.mnse import nmse_func
from src.utils.plots import (
    plot_distribution,
    plot_generated_image,
    plot_loss_func,
    plot_nmse_func,
    plot_skewness_func,
    plot_vae_in_out,
    show_loss_images,
    show_tensor_images,
)
from src.utils.utils import compute_mean_std, load_config, save_checkpoint, save_npy


def channel_est_vae_multi_antenna(vae, train_dataset, eval_dataset, n_epochs, device):
    save_results = "channel_est_based_vae_multi_antenna"
    min_nmse_scores = 10000000
    total_g_losses = []
    total_c_losses = []
    total_nmse_scores = []
    for epoch in range(n_epochs):
        # Training
        g_losses = 0
        nmse_scores = 0

        vae.train()
        for batch_index, (real, label) in enumerate(train_dataset):

            real = real.to(device)
            label = label.to(device)

            vae_opt.zero_grad()

            # get noise
            # Forward pass
            fake, *values = vae(label)

            # Compute loss
            v_loss, *values = vae_loss(fake, real, STD_H, MU_H, *values)

            g_loss = v_loss

            # Update gradients
            g_loss.backward()

            # Update the weights
            vae_opt.step()

            g_losses += g_loss.detach().cpu().item()

            if batch_index % (len(train_dataset) // 3) == 0:
                print(
                    f"Batch [{batch_index}/{len(train_dataset)}] - vae loss: ",
                    g_losses / (batch_index + 1),
                )

            nmse_score = nmse_func(
                pred=fake.permute(0, 2, 3, 1).detach().cpu(),
                target=real.permute(0, 2, 3, 1).detach().cpu(),
            )
            nmse_scores += nmse_score

        total_nmse_scores.append(nmse_scores / len(train_dataset))
        total_g_losses.append(g_losses / len(train_dataset))

        print(f"\tEpoch {epoch} - vae loss: {total_g_losses[-1]}")

        vae.eval()
        # Visualize Test Dataset
        with torch.no_grad():
            for real, label in eval_dataset:
                real = real.to(device)
                label = label.to(device)
                fake, *values = vae(label)
                _, _, (input_decoder, output_encoder) = values
                break
        if epoch in range(0, n_epochs, int(n_epochs // 10)):
            show_tensor_images(label[0, 0, :, :].T)
            show_tensor_images(fake[0, 0, :, :].T)
            show_tensor_images(real[0, 0, :, :].T)
            show_loss_images(total_c_losses, total_g_losses)

            plot_vae_in_out(label, fake, input_decoder, output_encoder, real)

        if min_nmse_scores > min(total_nmse_scores):
            min_nmse_scores = min(total_nmse_scores)
            save_checkpoint(vae, f"results/checkpoint/{save_results}/g_model.pt")
            plot_generated_image(
                fake[0, 0, :, :].T,
                test_input=label[0, 0, :, :].T,
                tar=real[0, 0, :, :].T,
                prefix=save_results,
            )
            print(f"\tMNSE Score: {min_nmse_scores}")

        plot_nmse_func(
            total_nmse_scores,
            range(epoch + 1),
            img_path=f"results/nmse/{save_results}/nmse.png",
        )
        plot_loss_func(
            None,
            total_g_losses,
            range(epoch + 1),
            img_path=f"results/loss/{save_results}/loss.png",
        )

    save_npy(
        f"results/nmse/{save_results}/nmse_scores.npy", np.array(total_nmse_scores)
    )
    save_npy(f"results/loss/{save_results}/losses_d.npy", np.array(total_c_losses))
    save_npy(f"results/loss/{save_results}/losses_g.npy", np.array(total_g_losses))


def channel_est_wgan_gp_vae_skew_multi_antenna(
    vae, train_dataset, eval_dataset, n_epochs, device
):
    save_results = "channel_est_wgan_gp_vae_skew_multi_antenna"
    min_nmse_scores = 10000000
    total_g_losses = []
    total_c_losses = []
    total_nmse_scores = []
    total_skewness = []
    for epoch in range(n_epochs):
        # Training
        g_losses = 0
        c_losses = 0
        g_skewness = 0
        nmse_scores = 0
        total_critic_reals = []
        total_critic_fakes = []
        total_generator_fakes = []

        vae.train()
        critic.train()
        for batch_index, (real, label) in enumerate(train_dataset):

            real = real.to(device)
            label = label.to(device)
            #################
            # Update critic #
            #################
            total_c_repeat_losses = 0
            critic_reals = 0
            critic_fakes = 0
            for _ in range(critic_repeats):
                c_opt.zero_grad()

                # Forward pass
                fake, *values = vae(label)

                # critic loss
                critic_real = critic(real)
                critic_fake = critic(fake)

                c_repeat_loss = -(torch.mean(critic_real) - torch.mean(critic_fake))

                # gp loss
                gp_loss = gradient_penalty_loss(critic=critic, real=real, fake=fake)

                # total critic loss
                total_c_repeat_loss = c_repeat_loss + c_lambda * gp_loss
                # Update gradients
                total_c_repeat_loss.backward(retain_graph=True)

                # Update optimizer
                c_opt.step()

                # Keep track of the average critic loss in this batch
                critic_reals += critic_real.detach().cpu()
                critic_fakes += critic_fake.detach().cpu()

                total_c_repeat_losses += total_c_repeat_loss.detach().cpu().item()

            total_critic_reals.append((critic_reals / critic_repeats).view(-1))
            total_critic_fakes.append((critic_fakes / critic_repeats).view(-1))
            c_losses += total_c_repeat_losses / critic_repeats

            ####################
            # Update generator #
            ####################
            vae_opt.zero_grad()

            # get noise
            # Forward pass
            fake_2, *values = vae(label)

            # Compute loss
            v_loss, *values = vae_loss(fake_2, real, STD_H, MU_H, *values)

            critic_fake_2 = critic(fake_2)

            skew_loss_2 = l2_loss(
                skew_cal(real.view(real.size(0), -1)),
                skew_cal(fake_2.view(fake_2.size(0), -1)),
            )

            g_loss = -torch.mean(critic_fake_2) + skew_loss_2 + v_loss

            # Update gradients
            g_loss.backward()

            # Update the weights
            vae_opt.step()

            g_losses += g_loss.detach().cpu().item()
            total_generator_fakes.append(critic_fake_2.detach().cpu().view(-1))
            g_skewness += skew_loss_2.detach().cpu().item()

            if batch_index % (len(train_dataset) // 3) == 0:
                print(
                    f"Batch [{batch_index}/{len(train_dataset)}] - ",
                    "Critic loss: ",
                    c_losses / (batch_index + 1),
                    "- generator loss: ",
                    g_losses / (batch_index + 1),
                )

            nmse_score = nmse_func(
                pred=fake.permute(0, 2, 3, 1).detach().cpu(),
                target=real.permute(0, 2, 3, 1).detach().cpu(),
            )
            nmse_scores += nmse_score

        total_nmse_scores.append(nmse_scores / len(train_dataset))
        total_c_losses.append(c_losses / len(train_dataset))
        total_g_losses.append(g_losses / len(train_dataset))
        total_skewness.append(g_skewness / len(train_dataset))

        print(
            f"\tEpoch {epoch} - critic loss: {total_c_losses[-1]} - generator loss: {total_g_losses[-1]}"
        )

        vae.eval()
        critic.eval()
        # Visualize Test Dataset
        with torch.no_grad():
            for real, label in eval_dataset:
                real = real.to(device)
                label = label.to(device)
                fake, *values = vae(label)
                _, _, (input_decoder, output_encoder) = values
                break
        if epoch in range(0, n_epochs, int(n_epochs // 10)):
            show_tensor_images(label[0, 0, :, :].T)
            show_tensor_images(fake[0, 0, :, :].T)
            show_tensor_images(real[0, 0, :, :].T)
            show_loss_images(total_c_losses, total_g_losses)

            plot_distribution(
                np.array(torch.cat(total_critic_reals)),
                np.array(torch.cat(total_critic_fakes)),
                np.array(torch.cat(total_generator_fakes)),
                save_path=f"results/dis/{save_results}/dis_{epoch}.png",
            )
            plot_vae_in_out(label, fake, input_decoder, output_encoder, real)

            grad_cam = GradCam(critic, critic.critic[-3][0])
            grad_cam_1 = GradCam(critic, critic.critic[-2][0])
            grad_cam_2 = GradCam(critic, critic.critic[-1][0])
            grad_cam.visualize_cam(
                fake,
                save_path=f"results/dam/{save_results}/grad_cam_fake_conv_1_epoch{epoch}.png",
            )
            grad_cam_1.visualize_cam(
                fake,
                save_path=f"results/dam/{save_results}/grad_cam_fake_conv_2_epoch{epoch}.png",
            )
            grad_cam_2.visualize_cam(
                fake,
                save_path=f"results/dam/{save_results}/grad_cam_fake_conv_3_epoch{epoch}.png",
            )

        if min_nmse_scores > min(total_nmse_scores):
            min_nmse_scores = min(total_nmse_scores)
            save_checkpoint(vae, f"results/checkpoint/{save_results}/g_model.pt")
            save_checkpoint(critic, f"results/checkpoint/{save_results}/d_model.pt")
            plot_generated_image(
                fake[0, 0, :, :].T,
                test_input=label[0, 0, :, :].T,
                tar=real[0, 0, :, :].T,
                prefix=save_results,
            )
            print(f"\tMNSE Score: {min_nmse_scores}")

        plot_nmse_func(
            total_nmse_scores,
            range(epoch + 1),
            img_path=f"results/nmse/{save_results}/nmse.png",
        )
        plot_loss_func(
            total_c_losses,
            total_g_losses,
            range(epoch + 1),
            img_path=f"results/loss/{save_results}/loss.png",
        )

        plot_skewness_func(
            total_skewness,
            range(epoch + 1),
            img_path=f"results/skewness/{save_results}/skewness.png",
        )

    save_npy(
        f"results/nmse/{save_results}/nmse_scores.npy", np.array(total_nmse_scores)
    )
    save_npy(f"results/loss/{save_results}/losses_d.npy", np.array(total_c_losses))
    save_npy(f"results/loss/{save_results}/losses_g.npy", np.array(total_g_losses))


def channel_est_wgan_gp_vae_multi_antenna(
    vae, train_dataset, eval_dataset, n_epochs, device
):
    save_results = "channel_est_wgan_gp_vae_multi_antenna"
    min_nmse_scores = 10000000
    total_g_losses = []
    total_c_losses = []
    total_nmse_scores = []
    for epoch in range(n_epochs):
        # Training
        g_losses = 0
        c_losses = 0
        nmse_scores = 0

        total_critic_reals = []
        total_critic_fakes = []
        total_generator_fakes = []

        vae.train()
        critic.train()
        for batch_index, (real, label) in enumerate(train_dataset):

            real = real.to(device)
            label = label.to(device)
            #################
            # Update critic #
            #################
            total_c_repeat_losses = 0
            critic_reals = 0
            critic_fakes = 0
            for _ in range(critic_repeats):
                c_opt.zero_grad()

                # Forward pass
                fake, *values = vae(label)

                # critic loss
                critic_real = critic(real)
                critic_fake = critic(fake)

                c_repeat_loss = -(torch.mean(critic_real) - torch.mean(critic_fake))

                # gp loss
                gp_loss = gradient_penalty_loss(critic=critic, real=real, fake=fake)

                # total critic loss
                total_c_repeat_loss = c_repeat_loss + c_lambda * gp_loss
                # Update gradients
                total_c_repeat_loss.backward(retain_graph=True)

                # Update optimizer
                c_opt.step()

                # Keep track of the average critic loss in this batch
                critic_reals += critic_real.detach().cpu()
                critic_fakes += critic_fake.detach().cpu()

                total_c_repeat_losses += total_c_repeat_loss.detach().cpu().item()

            total_critic_reals.append((critic_reals / critic_repeats).view(-1))
            total_critic_fakes.append((critic_fakes / critic_repeats).view(-1))
            c_losses += total_c_repeat_losses / critic_repeats

            ####################
            # Update generator #
            ####################
            vae_opt.zero_grad()

            # get noise
            # Forward pass
            fake_2, *values = vae(label)

            # Compute loss
            v_loss, *values = vae_loss(fake_2, real, STD_H, MU_H, *values)

            critic_fake_2 = critic(fake_2)

            g_loss = -torch.mean(critic_fake_2) + v_loss

            # Update gradients
            g_loss.backward()

            # Update the weights
            vae_opt.step()

            g_losses += g_loss.detach().cpu().item()
            total_generator_fakes.append(critic_fake_2.detach().cpu().view(-1))

            if batch_index % (len(train_dataset) // 3) == 0:
                print(
                    f"Batch [{batch_index}/{len(train_dataset)}] - ",
                    "Critic loss: ",
                    c_losses / (batch_index + 1),
                    "- generator loss: ",
                    g_losses / (batch_index + 1),
                )

            nmse_score = nmse_func(
                pred=fake.permute(0, 2, 3, 1).detach().cpu(),
                target=real.permute(0, 2, 3, 1).detach().cpu(),
            )
            nmse_scores += nmse_score

        total_nmse_scores.append(nmse_scores / len(train_dataset))
        total_c_losses.append(c_losses / len(train_dataset))
        total_g_losses.append(g_losses / len(train_dataset))

        print(
            f"\tEpoch {epoch} - critic loss: {total_c_losses[-1]} - generator loss: {total_g_losses[-1]}"
        )

        vae.eval()
        critic.eval()
        # Visualize Test Dataset
        with torch.no_grad():
            for real, label in eval_dataset:
                real = real.to(device)
                label = label.to(device)
                fake, *values = vae(label)
                _, _, (input_decoder, output_encoder) = values
                break
        if epoch in range(0, n_epochs, int(n_epochs // 10)):
            show_tensor_images(label[0, 0, :, :].T)
            show_tensor_images(fake[0, 0, :, :].T)
            show_tensor_images(real[0, 0, :, :].T)
            show_loss_images(total_c_losses, total_g_losses)

            plot_distribution(
                np.array(torch.cat(total_critic_reals)),
                np.array(torch.cat(total_critic_fakes)),
                np.array(torch.cat(total_generator_fakes)),
                save_path=f"results/dis/{save_results}/dis_{epoch}.png",
            )
            plot_vae_in_out(label, fake, input_decoder, output_encoder, real)

            grad_cam = GradCam(critic, critic.critic[-3][0])
            grad_cam_1 = GradCam(critic, critic.critic[-2][0])
            grad_cam_2 = GradCam(critic, critic.critic[-1][0])
            grad_cam.visualize_cam(
                fake,
                save_path=f"results/dam/{save_results}/grad_cam_fake_conv_1_epoch{epoch}.png",
            )
            grad_cam_1.visualize_cam(
                fake,
                save_path=f"results/dam/{save_results}/grad_cam_fake_conv_2_epoch{epoch}.png",
            )
            grad_cam_2.visualize_cam(
                fake,
                save_path=f"results/dam/{save_results}/grad_cam_fake_conv_3_epoch{epoch}.png",
            )

        if min_nmse_scores > min(total_nmse_scores):
            min_nmse_scores = min(total_nmse_scores)
            save_checkpoint(vae, f"results/checkpoint/{save_results}/g_model.pt")
            save_checkpoint(critic, f"results/checkpoint/{save_results}/d_model.pt")
            plot_generated_image(
                fake[0, 0, :, :].T,
                test_input=label[0, 0, :, :].T,
                tar=real[0, 0, :, :].T,
                prefix=save_results,
            )
            print(f"\tMNSE Score: {min_nmse_scores}")

        plot_nmse_func(
            total_nmse_scores,
            range(epoch + 1),
            img_path=f"results/nmse/{save_results}/nmse.png",
        )
        plot_loss_func(
            total_c_losses,
            total_g_losses,
            range(epoch + 1),
            img_path=f"results/loss/{save_results}/loss.png",
        )

    save_npy(
        f"results/nmse/{save_results}/nmse_scores.npy", np.array(total_nmse_scores)
    )
    save_npy(f"results/loss/{save_results}/losses_d.npy", np.array(total_c_losses))
    save_npy(f"results/loss/{save_results}/losses_g.npy", np.array(total_g_losses))


if __name__ == "__main__":
    # load config
    config = load_config("src/configs/configs.yml")
    device = config["device"]
    lr = config["lr"]
    betas = config["betas"]
    batch_size = config["batch_size"]
    n_workers = config["n_workers"]
    epoches = config["epoches"]
    critic_repeats = config["critic_repeats"]
    c_lambda = config["c_lambda"]
    z_dim = config["z_dim"]

    # get the data
    channels, responses, transmits = topology_20000_samples(
        directory=config["datasets"]
    )

    dataloader = ChannelDataloader(
        y_res=responses,
        channels_res=channels,
        batch_size=batch_size,
        num_worker=n_workers,
    )
    train_dataset = dataloader.train_dataloader()
    eval_dataset = dataloader.eval_dataloader()

    # get mean and std of the channel on training set
    MU_H, STD_H = compute_mean_std(train_dataset.dataset.channels_res, axis=0)
    MU_H = torch.from_numpy(MU_H.reshape(-1)).float().to(device)
    STD_H = torch.from_numpy(STD_H.reshape(-1)).float().to(device)

    is_model = "vae_wgan_gp_skewness"

    if is_model == "vae":
        vae = VAE(z_dim=z_dim, mu_h=MU_H, std_h=STD_H).to(device)

        # Initialize weights
        vae = vae.apply(weights_init)

        # Optimizer is Adam help converge faster and more stable
        vae_opt = torch.optim.Adam(vae.parameters(), lr=lr, betas=betas)

        channel_est_vae_multi_antenna(vae, train_dataset, eval_dataset, epoches, device)

    if is_model == "vae_wgan_gp":
        vae = VAE(z_dim=z_dim, mu_h=MU_H, std_h=STD_H).to(device)
        critic = Critic().to(device)

        # Initialize weights
        vae = vae.apply(weights_init)
        critic = critic.apply(weights_init)

        # Optimizer is Adam help converge faster and more stable
        vae_opt = torch.optim.Adam(vae.parameters(), lr=lr, betas=betas)
        c_opt = torch.optim.Adam(critic.parameters(), lr=lr, betas=betas)

        skew_cal = SkewCalculator(device)

        channel_est_wgan_gp_vae_multi_antenna(
            vae, train_dataset, eval_dataset, epoches, device
        )

    if is_model == "vae_wgan_gp_skewness":
        vae = VAE(z_dim=z_dim, mu_h=MU_H, std_h=STD_H).to(device)
        critic = Critic().to(device)

        # Initialize weights
        vae = vae.apply(weights_init)
        critic = critic.apply(weights_init)

        # Optimizer is Adam help converge faster and more stable
        vae_opt = torch.optim.Adam(vae.parameters(), lr=lr, betas=betas)
        c_opt = torch.optim.Adam(critic.parameters(), lr=lr, betas=betas)

        skew_cal = SkewCalculator(device)

        channel_est_wgan_gp_vae_skew_multi_antenna(
            vae, train_dataset, eval_dataset, epoches, device
        )
