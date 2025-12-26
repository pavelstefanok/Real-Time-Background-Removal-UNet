import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()

        self.enc1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.bottleneck = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # 128 canale la intrare deoarece primeste 64 de la up1 + 64 de la enc3
        self.dec1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        # 64 canale la intrare: 32 de la up2 + 32 de la enc2
        self.dec2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        # 32 canale la intrare: 16 de la up3 + 16 de la enc1
        self.dec3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)

        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # Encoder - salvam s1, s2, s3 pentru Skip Connections
        s1 = F.relu(self.enc1(x))
        s2 = F.relu(self.enc2(self.pool(s1)))
        s3 = F.relu(self.enc3(self.pool(s2)))

        # Bottleneck
        b = F.relu(self.bottleneck(self.pool(s3)))

        # Decoder
        u1 = self.up1(b)
        u1 = torch.cat([u1, s3], dim=1) #SC1
        u1 = F.relu(self.dec1(u1))

        u2 = self.up2(u1)
        u2 = torch.cat([u2, s2], dim=1) #2
        u2 = F.relu(self.dec2(u2))

        u3 = self.up3(u2)
        u3 = torch.cat([u3, s1], dim=1) #3
        u3 = F.relu(self.dec3(u3))

        return torch.sigmoid(self.out_conv(u3))

if __name__ == "__main__":
    model = SimpleUNet()
    dummy_input = torch.randn(1, 3, 256, 256)
    print("Output shape:", model(dummy_input).shape)