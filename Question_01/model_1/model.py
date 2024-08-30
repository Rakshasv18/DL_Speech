import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Initialize the UNet model.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(UNet, self).__init__()
        
        # Encoder
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.decoder4 = self.upconv_block(1024, 512)
        self.decoder3 = self.upconv_block(512, 256)
        self.decoder2 = self.upconv_block(256, 128)
        self.decoder1 = self.upconv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """
        Create a convolutional block with two convolutional layers followed by ReLU activation.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Returns:
            nn.Sequential: Sequential model containing the convolutional layers and activation functions.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def upconv_block(self, in_channels, out_channels):
        """
        Create an upsampling block with a transposed convolution layer followed by ReLU activation.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Returns:
            nn.Sequential: Sequential model containing the upsampling layer and activation function.
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Perform the forward pass through the UNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width) after passing through the network.
        """
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder
        d4 = self.decoder4(b)
        d4 = torch.cat([d4, e4], dim=1)  # Concatenate along channel dimension
        d3 = self.decoder3(d4)
        d3 = torch.cat([d3, e3], dim=1)  # Concatenate along channel dimension
        d2 = self.decoder2(d3)
        d2 = torch.cat([d2, e2], dim=1)  # Concatenate along channel dimension
        d1 = self.decoder1(d2)
        d1 = torch.cat([d1, e1], dim=1)  # Concatenate along channel dimension
        
        return self.final_conv(d1)
