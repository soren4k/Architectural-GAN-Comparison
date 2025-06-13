import torch
import torch.nn as nn

# Standard weight initialization for DC-GAN and WGAN-GP
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator remains identical to DC-GAN's design
class Generator(nn.Module):
    def __init__(self, latent_dim, feature_maps_g, num_channels):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_maps_g * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps_g * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps_g * 16, feature_maps_g * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_g * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps_g * 8, feature_maps_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_g * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps_g * 4, feature_maps_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_g * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps_g * 2, feature_maps_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_g),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps_g, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# WGAN-GP Critic without any BatchNorm layers
class Critic(nn.Module):
    def __init__(self, num_channels, feature_maps_d):
        super(Critic, self).__init__()
        self.main = nn.Sequential(
            # 128x128 -> 64x64
            nn.Conv2d(num_channels, feature_maps_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 64x64 -> 32x32
            nn.Conv2d(feature_maps_d, feature_maps_d * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(feature_maps_d * 2, feature_maps_d * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 8x8
            nn.Conv2d(feature_maps_d * 4, feature_maps_d * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 4x4
            nn.Conv2d(feature_maps_d * 8, feature_maps_d * 16, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 4x4 -> 1x1 score
            nn.Conv2d(feature_maps_d * 16, 1, 4, 1, 0, bias=False)
        )

    def forward(self, x):
        return self.main(x)


# --- [NEW] Critic with Instance Normalization for Stage 3 ---
class Critic_InstanceNorm(nn.Module):
    """
    A WGAN-GP Critic that uses Instance Normalization.
    """
    def __init__(self, num_channels, feature_maps_d):
        super(Critic_InstanceNorm, self).__init__()
        self.main = nn.Sequential(
            # 128x128 -> 64x64
            nn.Conv2d(num_channels, feature_maps_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 64x64 -> 32x32
            nn.Conv2d(feature_maps_d, feature_maps_d * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(feature_maps_d * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(feature_maps_d * 2, feature_maps_d * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(feature_maps_d * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 8x8
            nn.Conv2d(feature_maps_d * 4, feature_maps_d * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(feature_maps_d * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 4x4
            nn.Conv2d(feature_maps_d * 8, feature_maps_d * 16, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(feature_maps_d * 16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 4x4 -> 1x1 score
            nn.Conv2d(feature_maps_d * 16, 1, 4, 1, 0, bias=False)
        )

    def forward(self, x):
        return self.main(x)