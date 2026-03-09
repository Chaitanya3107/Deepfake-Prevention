import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)

class UNetGen(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = DoubleConv(base * 4 + base * 2, base * 2)
        self.dec2 = DoubleConv(base * 2 + base, base)
        self.out = nn.Conv2d(base, in_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d3 = self.dec3(torch.cat([self.up(e3), e2], 1))
        d2 = self.dec2(torch.cat([self.up(d3), e1], 1))
        out = torch.tanh(self.out(d2))
        return (out + 1) / 2

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(base, base*2, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(base*2, base*4, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(base*4, 1, 4, 1, 1),
        )
    def forward(self, x):
        return self.net(x)