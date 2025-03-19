import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim=100, im_chan=2, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

        self.transform = nn.Sequential(
            nn.Linear(z_dim, hidden_dim * 4 * 4),
            nn.ReLU(),
        )

        self.gen = nn.Sequential(
            self.make_gen_block(
                hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1
            ),
            self.make_gen_block(
                hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1
            ),
            self.make_gen_block(
                hidden_dim // 4,
                im_chan,
                kernel_size=4,
                stride=2,
                padding=1,
                final_layer=True,
            ),
        )

    def make_gen_block(
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
                nn.Tanh(),
            )

    def forward(self, noise):
        x = self.transform(noise)
        x = x.view(len(x), self.hidden_dim, 4, 4)
        return self.gen(x)


class Discriminator(nn.Module):
    def __init__(self, im_chan=2, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.critic = nn.Sequential(
            self.make_critic_block(
                im_chan, hidden_dim, kernel_size=4, stride=2, padding=1
            ),  # 2x32x32 -> 64x16x16
            self.make_critic_block(
                hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1
            ),  # 64x16x16 -> 128x8x8
            self.make_critic_block(
                hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1
            ),  # 128x8x8 -> 256x4x4
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_dim * 4, 1)
        self.activation = nn.Sigmoid()

    def make_critic_block(
        self, input_channels, output_channels, kernel_size=4, stride=2, padding=1
    ):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, image):
        critic_pred = self.critic(image)
        critic_pred = self.avg_pool(critic_pred)
        critic_pred = critic_pred.view(critic_pred.size(0), -1)
        critic_pred = self.fc(critic_pred)
        critic_pred = self.activation(critic_pred)
        return critic_pred


class Critic(Discriminator):
    def __init__(self, im_chan=2, hidden_dim=64):
        super(Critic, self).__init__(im_chan, hidden_dim)

    def forward(self, image):
        critic_pred = self.critic(image)
        critic_pred = self.avg_pool(critic_pred)
        critic_pred = critic_pred.view(critic_pred.size(0), -1)
        critic_pred = self.fc(critic_pred)
        return critic_pred


if __name__ == "__main__":
    from torchsummary import summary

    g_model = Generator(32 * 32 * 2)

    summary(g_model, (32 * 32 * 2,))
