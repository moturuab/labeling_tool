import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN
# ----------------------------------------------------------------------------------------------------------------------


class CNN2D(nn.Module):

    def __init__(self, kernel_size=3):
        super(CNN2D, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 128, kernel_size),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.network(x)

# U-Net
# ----------------------------------------------------------------------------------------------------------------------
# Source: https://github.com/milesial/Pytorch-UNet
# Paper: https://arxiv.org/abs/1505.04597


class UNet2D(nn.Module):

    def __init__(self, n_channels=1, n_classes=2, bilinear=True):
        super(UNet2D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes * n_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# AnoGAN
# ----------------------------------------------------------------------------------------------------------------------
# Source: https://github.com/seungjunlee96/AnoGAN-pytorch
# Paper: https://arxiv.org/pdf/1703.05921.pdf


class AnoGANDiscrminator2D(nn.Module):

    def __init__(self, c_dim, df_dim):
        super(AnoGANDiscrminator2D, self).__init__()

        self.conv0 = nn.Conv2d(c_dim, df_dim, 4, 2, 1, bias=False)
        self.elu0 = nn.ELU(inplace=True)

        self.conv1 = nn.Conv2d(df_dim, df_dim * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(df_dim * 2)
        self.elu1 = nn.ELU(inplace=True)

        self.conv2 = nn.Conv2d(df_dim * 2, df_dim * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(df_dim * 4)
        self.elu2 = nn.ELU(inplace=True)

        self.conv3 = nn.Conv2d(df_dim * 4, df_dim * 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(df_dim * 8)
        self.elu3 = nn.ELU(inplace=True)

        self.conv4 = nn.Conv2d(df_dim * 8, df_dim * 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(df_dim * 16)
        self.elu4 = nn.ELU(inplace=True)

        self.conv5 = nn.Conv2d(df_dim * 16, 1, kernel_size=4, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(mean=0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, inp):
        h0 = self.elu0(self.conv0(inp))
        h1 = self.elu1(self.bn1(self.conv1(h0)))
        h2 = self.elu2(self.bn2(self.conv2(h1)))
        h3 = self.elu3(self.bn3(self.conv3(h2)))
        h4 = self.elu4(self.bn4(self.conv4(h3)))
        h5 = self.conv5(h4)
        out = self.sigmoid(h5)
        return h4, out.view(-1, 1)  # by squeeze get just not float Tensor


class AnoGANGenerator2D(nn.Module):

    def __init__(self, z_dim, gf_dim):
        super(AnoGANGenerator2D, self).__init__()

        # torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
        #                          output_padding=0,groups=1, bias=True, dilation=1, padding_mode='zeros')

        self.trans0 = nn.ConvTranspose2d(z_dim, gf_dim * 16, 4, 1, 0,
                                         bias=False)  # no bias term when applying Batch Normalization
        self.bn0 = nn.BatchNorm2d(gf_dim * 16)
        self.elu0 = nn.ELU(inplace=True)

        self.trans1 = nn.ConvTranspose2d(in_channels=gf_dim * 16, out_channels=gf_dim * 8, kernel_size=4,
                                         stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(gf_dim * 8)
        self.elu1 = nn.ELU(inplace=True)

        self.trans2 = nn.ConvTranspose2d(in_channels=gf_dim * 8, out_channels=gf_dim * 4, kernel_size=4,
                                         stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(gf_dim * 4)
        self.elu2 = nn.ELU(inplace=True)

        self.trans3 = nn.ConvTranspose2d(in_channels=gf_dim * 4, out_channels=gf_dim * 2, kernel_size=4,
                                         stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(gf_dim * 2)
        self.elu3 = nn.ELU(inplace=True)

        self.trans4 = nn.ConvTranspose2d(in_channels=gf_dim * 2, out_channels=gf_dim, kernel_size=4,
                                         stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(gf_dim)
        self.elu4 = nn.ELU(inplace=True)

        self.trans5 = nn.ConvTranspose2d(in_channels=gf_dim, out_channels=1, kernel_size=4,
                                         stride=2, padding=1, bias=False)
        self.tanh = nn.Tanh()

        for module in self.modules():
            if isinstance(module, nn.ConvTranspose2d):
                module.weight.data.normal_(mean=0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, inp):
        h0 = self.elu0(self.bn0(self.trans0(inp)))
        h1 = self.elu1(self.bn1(self.trans1(h0)))
        h2 = self.elu2(self.bn2(self.trans2(h1)))
        h3 = self.elu3(self.bn3(self.trans3(h2)))
        h4 = self.elu4(self.bn4(self.trans4(h3)))
        h5 = self.trans5(h4)
        out = self.tanh(h5)

        return out

# BetaVAE
# ----------------------------------------------------------------------------------------------------------------------
# Source: https://github.com/matthew-liu/beta-vae
# Paper: https://openreview.net/forum?id=Sy2fzU9gl


class BetaVAE2D(nn.Module):

    def __init__(self, latent_size=32, beta=1):
        super(BetaVAE2D, self).__init__()

        self.latent_size = latent_size
        self.beta = beta

        # encoder
        self.encoder = nn.Sequential(
            self._conv(1, 32),
            self._conv(32, 32),
            self._conv(32, 64),
            self._conv(64, 128),
            self._conv(128, 128),
        )
        self.fc_mu = nn.Linear(256, latent_size)
        self.fc_var = nn.Linear(256, latent_size)

        # decoder
        self.decoder = nn.Sequential(
            self._deconv(128, 128),
            self._deconv(128, 64),
            self._deconv(64, 32),
            self._deconv(32, 32, 1),
            self._deconv(32, 1),
            nn.Sigmoid()
        )
        self.fc_z = nn.Linear(latent_size, 256)

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(-1, 256)
        return self.fc_mu(x), self.fc_var(x)

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # e^(1/2 * log(std^2))
        eps = torch.randn_like(std)  # random ~ N(0, 1)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = self.fc_z(z)
        z = z.view(-1, 128, 2, 2)
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        rx = self.decode(z)
        return rx, mu, logvar

    def _conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=4, stride=2
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    # out_padding is used to ensure output size matches EXACTLY of conv2d;
    # it does not actually add zero-padding to output :)
    def _deconv(self, in_channels, out_channels, out_padding=0):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, output_padding=out_padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def loss(self, recon_x, x, mu, logvar):
        # reconstruction losses are summed over all elements and batch
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_diverge = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return (recon_loss + self.beta * kl_diverge) / x.shape[0]  # divide total loss by batch size