import torch
import torch.nn as nn
import torch.nn.functional as F


class OriginalVAE(nn.Module):
    def __init__(self, z_dim=256, im_chan=2):
        super(OriginalVAE, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = int(z_dim // 8)

        # Encoder
        self.encoder = nn.Sequential(
            self.make_encoder_block(
                im_chan, self.hidden_dim * 2, kernel_size=4, stride=2, padding=1
            ),  # (2, 8, 32) -> (hidden * 2, 4, 16)
            self.make_encoder_block(
                self.hidden_dim * 2,
                self.hidden_dim * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),  # (hidden * 2, 4, 16) -> (hidden * 4, 2, 8)
            self.make_encoder_block(
                self.hidden_dim * 4,
                self.hidden_dim * 8,
                kernel_size=4,
                stride=2,
                padding=1,
            ),  # (hidden * 4, 2, 8) -> (hidden * 4, 1, 4)
            self.make_encoder_block(
                self.hidden_dim * 8,
                self.hidden_dim * 8,
                kernel_size=4,
                stride=2,
                padding=0,
            ),  # (hidden * 4, 1, 4) -> (hidden * 4, 1, 1)
        )
        self.fc_mu = nn.Linear(self.hidden_dim * 8, z_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim * 8, z_dim)

        # Decoder
        self.decoder = nn.Sequential(
            self.make_decoder_block(
                self.hidden_dim * 8,
                self.hidden_dim * 8,
                kernel_size=4,
                stride=2,
                padding=0,
            ),  # (hidden * 4, 1, 1) -> (hidden * 4, 1, 4)
            self.make_decoder_block(
                self.hidden_dim * 8,
                self.hidden_dim * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),  # (hidden * 4, 1, 4) -> (hidden * 4, 2, 8)
            self.make_decoder_block(
                self.hidden_dim * 4,
                self.hidden_dim * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),  # (hidden * 4, 2, 8) -> (hidden * 2, 4, 16)
            self.make_decoder_block(
                self.hidden_dim * 2,
                im_chan,
                kernel_size=4,
                stride=2,
                padding=1,
                final_layer=True,
            ),  # (hidden * 2, 4, 16) -> (2, 8, 32)
        )

    def make_encoder_block(
        self,
        input_channels,
        output_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        final_layer=False,
    ):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def make_decoder_block(
        self,
        input_channels,
        output_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        final_layer=False,
    ):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    input_channels, output_channels, kernel_size, stride, padding
                ),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    input_channels, output_channels, kernel_size, stride, padding
                ),
                nn.Tanh(),  # Output in the range [-1, 1]
            )

    def encode(self, x):
        output_encoder = self.encoder(x)
        output_encoder = output_encoder.view(output_encoder.size(0), -1)
        return (
            self.fc_mu(output_encoder),
            self.fc_logvar(output_encoder),
            output_encoder,
        )

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        z_latent = mu + eps * std

        return z_latent

    def decode(self, z):
        h = z.view(z.size(0), -1, 1, 1)
        return self.decoder(h)

    def forward(self, x, *args):
        mu, logvar, output_encoder = self.encode(x)

        z = self.reparameterize(mu, logvar)

        return self.decode(z), mu, logvar, (z, output_encoder)


class VAE(nn.Module):
    def __init__(self, z_dim=256, im_chan=2, mu_h=None, std_h=None):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = int(z_dim // 8)
        self.mu_h = mu_h
        self.std_h = std_h

        # Encoder
        self.encoder = nn.Sequential(
            self.make_encoder_block(
                im_chan, self.hidden_dim * 2, kernel_size=4, stride=2, padding=1
            ),  # (2, 8, 32) -> (hidden * 2, 4, 16)
            self.make_encoder_block(
                self.hidden_dim * 2,
                self.hidden_dim * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),  # (hidden * 2, 4, 16) -> (hidden * 4, 2, 8)
            self.make_encoder_block(
                self.hidden_dim * 4,
                self.hidden_dim * 8,
                kernel_size=4,
                stride=2,
                padding=1,
            ),  # (hidden * 4, 2, 8) -> (hidden * 4, 1, 4)
            self.make_encoder_block(
                self.hidden_dim * 8,
                self.hidden_dim * 16,
                kernel_size=4,
                stride=2,
                padding=1,
            ),  # (hidden * 4, 2, 8) -> (hidden * 4, 1, 4)
        )

        # Decoder
        self.decoder = nn.Sequential(
            self.make_decoder_block(
                self.hidden_dim * 16,
                self.hidden_dim * 8,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            self.make_decoder_block(
                self.hidden_dim * 8,
                self.hidden_dim * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),  # (hidden * 4, 1, 4) -> (hidden * 4, 2, 8)
            self.make_decoder_block(
                self.hidden_dim * 4,
                self.hidden_dim * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),  # (hidden * 4, 2, 8) -> (hidden * 2, 4, 16)
            self.make_decoder_block(
                self.hidden_dim * 2,
                im_chan,
                kernel_size=4,
                stride=2,
                padding=1,
                final_layer=True,
            ),  # (hidden * 2, 4, 16) -> (2, 8, 32)
        )

    def make_encoder_block(
        self,
        input_channels,
        output_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        final_layer=False,
    ):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def make_decoder_block(
        self,
        input_channels,
        output_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        final_layer=False,
    ):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    input_channels, output_channels, kernel_size, stride, padding
                ),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    input_channels, output_channels, kernel_size, stride, padding
                ),
                nn.Tanh(),  # Output in the range [-1, 1]
            )

    def encode(self, x):
        output_encoder = self.encoder(x)
        output_encoder = output_encoder.view(output_encoder.size(0), -1)
        return output_encoder

    def reparameterize(self, output_encoder):
        if isinstance(self.mu_h, torch.Tensor) and isinstance(self.std_h, torch.Tensor):
            z_latent = self.mu_h + output_encoder * self.std_h
        else:
            z_latent = torch.randn(output_encoder.shape).cuda()

        return z_latent

    def decode(self, z):
        h = z.view(z.size(0), -1, 2, 2)
        return self.decoder(h)

    def forward(self, x, *args):
        output_encoder = self.encode(x)

        z = self.reparameterize(output_encoder)

        return self.decode(z), z, z, (z, output_encoder)


if __name__ == "__main__":
    vae = OriginalVAE(z_dim=256)

    x = torch.randn(1, 2, 8, 32)
