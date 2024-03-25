from torch import nn

# ----------------------------------------------------------------
# define the model
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),  # b, 16, 28, 28
            nn.ReLU(True),
            ResidualBlock(16),
            nn.Conv2d(16, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            ResidualBlock(16),
            nn.Conv2d(16, 12, 3, stride=1, padding=1),  # b, 12, 5, 5
            nn.ReLU(True),
            nn.Conv2d(12, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            ResidualBlock(8),
            nn.ConvTranspose2d(8, 12, 3, stride=2, padding=1),  # b, 12, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(12, 16, 2, stride=2),  # b, 16, 10, 10
            nn.ReLU(True),
            ResidualBlock(16),
            nn.ConvTranspose2d(16, 1, 3, stride=3, padding=1),  # b, 1, 28, 28
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded