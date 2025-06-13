"""
COGS 185 Final Project - DC-GAN Model Definitions
This script contains the Generator and Discriminator classes for the DC-GAN,
as well as the weights_init function.

The models are parameterized to allow for easy hyper-parameter tuning
from the main training notebook.
"""
import torch
import torch.nn as nn

# --- Custom Weight Initialization ---
def weights_init(m):
    """
    Initializes model weights to mean=0, stdev=0.02, as per the DC-GAN paper.
    This function can be applied to a model using model.apply(weights_init).
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # Initialize convolutional layers
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        # Initialize batch norm layers
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# --- Generator Network ---
class Generator(nn.Module):
    """
    DC-GAN Generator Network.
    Takes a latent vector Z and upsamples it to a 128x128 RGB image.
    
    Args:
        latent_dim (int): Size of the input latent vector (z).
        feature_maps_g (int): Base number of feature maps.
        num_channels (int): Number of output image channels (3 for RGB).
    """
    def __init__(self, latent_dim, feature_maps_g, num_channels):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input: Z (latent_dim x 1 x 1), going into a convolution
            nn.ConvTranspose2d(latent_dim, feature_maps_g * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps_g * 16),
            nn.ReLU(True),
            # State size: (feature_maps_g * 16) x 4 x 4

            nn.ConvTranspose2d(feature_maps_g * 16, feature_maps_g * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_g * 8),
            nn.ReLU(True),
            # State size: (feature_maps_g * 8) x 8 x 8

            nn.ConvTranspose2d(feature_maps_g * 8, feature_maps_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_g * 4),
            nn.ReLU(True),
            # State size: (feature_maps_g * 4) x 16 x 16

            nn.ConvTranspose2d(feature_maps_g * 4, feature_maps_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_g * 2),
            nn.ReLU(True),
            # State size: (feature_maps_g * 2) x 32 x 32

            nn.ConvTranspose2d(feature_maps_g * 2, feature_maps_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_g),
            nn.ReLU(True),
            # State size: (feature_maps_g) x 64 x 64

            nn.ConvTranspose2d(feature_maps_g, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Final state size: (num_channels) x 128 x 128
        )

    def forward(self, input):
        return self.main(input)

# --- Discriminator Network ---
class Discriminator(nn.Module):
    """
    DC-GAN Discriminator Network.
    Takes a 128x128 image and classifies it as real or fake.

    Args:
        num_channels (int): Number of input image channels (3 for RGB).
        feature_maps_d (int): Base number of feature maps.
    """
    def __init__(self, num_channels, feature_maps_d):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: (num_channels) x 128 x 128
            nn.Conv2d(num_channels, feature_maps_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (feature_maps_d) x 64 x 64

            nn.Conv2d(feature_maps_d, feature_maps_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (feature_maps_d * 2) x 32 x 32

            nn.Conv2d(feature_maps_d * 2, feature_maps_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (feature_maps_d * 4) x 16 x 16

            nn.Conv2d(feature_maps_d * 4, feature_maps_d * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (feature_maps_d * 8) x 8 x 8

            nn.Conv2d(feature_maps_d * 8, feature_maps_d * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_d * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (feature_maps_d * 16) x 4 x 4

            nn.Conv2d(feature_maps_d * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Final state size: 1x1x1 (a probability)
        )

    def forward(self, input):
        return self.main(input)

# --- [NEW] Deeper Generator for Stage 3 ---
class Generator_Deeper(nn.Module):
    """
    A deeper version of the DC-GAN Generator with an extra upsampling layer.
    Starts from a 2x2 spatial size instead of 4x4.
    """
    def __init__(self, latent_dim, feature_maps_g, num_channels):
        super(Generator_Deeper, self).__init__()
        self.main = nn.Sequential(
            # Start at 2x2
            nn.ConvTranspose2d(latent_dim, feature_maps_g * 32, 2, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps_g * 32),
            nn.ReLU(True),
            # State size: (feature_maps_g * 32) x 2 x 2
            
            nn.ConvTranspose2d(feature_maps_g * 32, feature_maps_g * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_g * 16),
            nn.ReLU(True),
            # State size: (feature_maps_g * 16) x 4 x 4

            nn.ConvTranspose2d(feature_maps_g * 16, feature_maps_g * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_g * 8),
            nn.ReLU(True),
            # State size: (feature_maps_g * 8) x 8 x 8

            nn.ConvTranspose2d(feature_maps_g * 8, feature_maps_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_g * 4),
            nn.ReLU(True),
            # State size: (feature_maps_g * 4) x 16 x 16

            nn.ConvTranspose2d(feature_maps_g * 4, feature_maps_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_g * 2),
            nn.ReLU(True),
            # State size: (feature_maps_g * 2) x 32 x 32

            nn.ConvTranspose2d(feature_maps_g * 2, feature_maps_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_g),
            nn.ReLU(True),
            # State size: (feature_maps_g) x 64 x 64

            nn.ConvTranspose2d(feature_maps_g, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Final state size: (num_channels) x 128 x 128
        )

    def forward(self, input):
        return self.main(input)

# --- [NEW] Deeper Discriminator for Stage 3 ---
class Discriminator_Deeper(nn.Module):
    """
    A deeper version of the DC-GAN Discriminator with an extra downsampling layer.
    Ends at a 2x2 spatial size before the final projection.
    """
    def __init__(self, num_channels, feature_maps_d):
        super(Discriminator_Deeper, self).__init__()
        self.main = nn.Sequential(
            # Input: (num_channels) x 128 x 128
            nn.Conv2d(num_channels, feature_maps_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (feature_maps_d) x 64 x 64

            nn.Conv2d(feature_maps_d, feature_maps_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (feature_maps_d * 2) x 32 x 32

            nn.Conv2d(feature_maps_d * 2, feature_maps_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (feature_maps_d * 4) x 16 x 16

            nn.Conv2d(feature_maps_d * 4, feature_maps_d * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (feature_maps_d * 8) x 8 x 8

            nn.Conv2d(feature_maps_d * 8, feature_maps_d * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_d * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (feature_maps_d * 16) x 4 x 4

            # --- ADDED NEW LAYER ---
            nn.Conv2d(feature_maps_d * 16, feature_maps_d * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_d * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (feature_maps_d * 32) x 2 x 2
            
            # Final projection from 2x2 to 1x1
            nn.Conv2d(feature_maps_d * 32, 1, 2, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# --- This block allows for testing the script directly ---
# It will only run when you execute `python models_dcgan.py`
# It will NOT run when you import it into your notebook.
if __name__ == '__main__':
    # --- Verification Step ---
    print("--- Running Direct Verification of models_dcgan.py ---")
    
    # Example parameters
    LATENT_DIM_TEST = 100
    FEATURES_G_TEST = 64
    FEATURES_D_TEST = 64
    NUM_CHANNELS_TEST = 3
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    # Test original models
    print("\n🧠 Initializing ORIGINAL models...")
    netG = Generator(LATENT_DIM_TEST, FEATURES_G_TEST, NUM_CHANNELS_TEST).to(device)
    netD = Discriminator(NUM_CHANNELS_TEST, FEATURES_D_TEST).to(device)
    print("✅ Original models initialized.")
    print(netG)

    # Test deeper models
    print("\n🧠 Initializing DEEPER models...")
    netG_deeper = Generator_Deeper(LATENT_DIM_TEST, FEATURES_G_TEST, NUM_CHANNELS_TEST).to(device)
    netD_deeper = Discriminator_Deeper(NUM_CHANNELS_TEST, FEATURES_D_TEST).to(device)
    print("✅ Deeper models initialized.")
    print(netG_deeper)