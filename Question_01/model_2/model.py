import torch.nn as nn

class MaskedAutoencoder(nn.Module):
    """
    A simple Masked Autoencoder model with an encoder and decoder.

    This autoencoder consists of:
    - An encoder that applies a convolutional layer followed by ReLU activation and max pooling.
    - A decoder that applies a transposed convolutional layer followed by a sigmoid activation.

    Attributes:
        encoder (nn.Sequential): Sequential container of layers for encoding input.
        decoder (nn.Sequential): Sequential container of layers for decoding the encoded features.
    """

    def __init__(self):
        """
        Initializes the MaskedAutoencoder model by defining the encoder and decoder.
        """
        super(MaskedAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor to be processed by the autoencoder.

        Returns:
            torch.Tensor: Output tensor after passing through the encoder and decoder.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
