import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv3x3 -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        self.enc1 = DoubleConv(3, 16)
        self.enc2 = DoubleConv(16, 32)
        self.enc3 = DoubleConv(32, 64)

        # Bottleneck
        self.bottleneck = DoubleConv(64, 128)

        # Decoder
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64, 32)

        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(32, 16)

        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))

        # Bottleneck
        b = self.bottleneck(self.pool(s3))

        # Decoder
        u1 = self.up1(b)
        u1 = torch.cat([u1, s3], dim=1)
        u1 = self.dec1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, s2], dim=1)
        u2 = self.dec2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, s1], dim=1)
        u3 = self.dec3(u3)

        return torch.sigmoid(self.out_conv(u3))

if __name__ == "__main__":
    model = SimpleUNet()
    dummy_input = torch.randn(1, 3, 256, 256)
    print("Output shape:", model(dummy_input).shape)