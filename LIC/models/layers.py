import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import CompressionModel
from compressai.layers import ResidualBlock, ResidualBlockWithStride, ResidualBlockUpsample, conv3x3, subpel_conv3x3, \
    GDN, AttentionBlock
from compressai.models.utils import conv, deconv


class GeneralAnalysisLayer(nn.Module):
    """The General Analysis Transform"""

    def __init__(self, N=192):
        super(GeneralAnalysisLayer, self).__init__()

        self.analysis_transform = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

    def forward(self, x):
        return self.analysis_transform(x)

class GeneralAnalysisLayerAttention(nn.Module):
    """The General Analysis Transform"""

    def __init__(self, N=192):
        super(GeneralAnalysisLayerAttention, self).__init__()

        self.analysis_transform = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
            AttentionBlock(N),
        )

    def forward(self, x):
        return self.analysis_transform(x)


class GeneralAnalysisLayerGDN(nn.Module):
    """The General Analysis Transform"""

    def __init__(self, N=192):
        super(GeneralAnalysisLayerGDN, self).__init__()

        self.analysis_transform = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, 320)
        )

    def forward(self, x):
        return self.analysis_transform(x)


class GeneralSynthesisLayer(nn.Module):
    """The General Analysis Transform"""

    def __init__(self, N=192):
        super(GeneralSynthesisLayer, self).__init__()

        self.synthesis_transform = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

    def forward(self, x):
        return self.synthesis_transform(x)


class GeneralSynthesisLayerAttention(nn.Module):
    """The General Analysis Transform"""

    def __init__(self, N=192):
        super(GeneralSynthesisLayerAttention, self).__init__()

        self.synthesis_transform = nn.Sequential(
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

    def forward(self, x):
        return self.synthesis_transform(x)


class GeneralSynthesisLayerGDN(nn.Module):
    """The General Analysis Transform"""

    def __init__(self, N=192):
        super(GeneralSynthesisLayerGDN, self).__init__()

        self.synthesis_transform = nn.Sequential(
            deconv(320, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3)
        )

    def forward(self, x):
        return self.synthesis_transform(x)


class HyperAnalysisLayer(nn.Module):
    """The analysis transform for the entropy model parameters"""

    def __init__(self, N=192):
        super(HyperAnalysisLayer, self).__init__()

        self.hyper_analysis_transform = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

    def forward(self, x):
        return self.hyper_analysis_transform(x)


class HyperAnalysisLayerGDN(nn.Module):
    """The analysis transform for the entropy model parameters"""

    def __init__(self, N=192):
        super(HyperAnalysisLayerGDN, self).__init__()

        self.hyper_analysis_transform = nn.Sequential(
            conv(320, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N)
        )

    def forward(self, x):
        return self.hyper_analysis_transform(x)


class HyperSynthesisLayer(nn.Module):
    """The synthesis transform for the entropy model parameters"""

    def __init__(self, N=192):
        super(HyperSynthesisLayer, self).__init__()

        self.hyper_synthesis_transform = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

    def forward(self, x):
        return self.hyper_synthesis_transform(x)


class HyperSynthesisLayerGDN(nn.Module):
    """The synthesis transform for the entropy model parameters"""

    def __init__(self, N=192):
        super(HyperSynthesisLayerGDN, self).__init__()

        self.hyper_synthesis_transform = nn.Sequential(
            deconv(N, 320),
            nn.LeakyReLU(inplace=True),
            deconv(320, 320 * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(320 * 3 // 2, 320 * 2, stride=1, kernel_size=3)
        )

    def forward(self, x):
        return self.hyper_synthesis_transform(x)


class PackAnalysisLayer(nn.Module):
    """The analysis transform for packing"""

    def __init__(self, r=4):
        super(PackAnalysisLayer, self).__init__()

        self.r = r
        self.pixel_shuffle = nn.PixelShuffle(r)

    def forward(self, x):
        x = self.pixel_shuffle(x)
        N, C, H, W = x.size()
        x_slices = torch.chunk(x, C, 1)
        packs = []

        for index, x_slice in enumerate(x_slices):
            if index % 2 == 0:
                temp = torch.chunk(x_slice, 4, 2)
                packs.append(torch.cat([temp[0], temp[2]], 2))
                packs.append(torch.cat([temp[1], temp[3]], 2))
            else:
                temp = torch.chunk(x_slice, 4, 3)
                packs.append(torch.cat([temp[0], temp[2]], 3))
                packs.append(torch.cat([temp[1], temp[3]], 3))

        return packs


class PackSynthesisLayer(nn.Module):
    """The synthesis transform for packing"""

    def __init__(self, r=4):
        super(PackSynthesisLayer, self).__init__()

        self.r = r
        self.pixel_un_shuffle = nn.PixelUnshuffle(r)

    def forward(self, x):
        C = len(x) // 2
        N, _, H, W = x[0].size()
        device = x[0].device
        H = H * 2
        h = H // 4
        w = W // 4
        y = torch.zeros([N, C, H, W]).to(device)

        for index, pack in enumerate(x):
            if (index // 2) % 2 == 0:
                temp = torch.chunk(pack, 2, 2)
                if index % 2 == 0:
                    y[:, index // 2, h * 0:h * 1, :] = temp[0][:, 0, :, :]
                    y[:, index // 2, h * 2:h * 3, :] = temp[1][:, 0, :, :]
                else:
                    y[:, index // 2, h * 1:h * 2, :] = temp[0][:, 0, :, :]
                    y[:, index // 2, h * 3:h * 4, :] = temp[1][:, 0, :, :]
            else:
                temp = torch.chunk(pack, 2, 3)
                if index % 2 == 0:
                    y[:, index // 2, :, w * 0:w * 1] = temp[0][:, 0, :, :]
                    y[:, index // 2, :, w * 2:w * 3] = temp[1][:, 0, :, :]
                else:
                    y[:, index // 2, :, w * 1:w * 2] = temp[0][:, 0, :, :]
                    y[:, index // 2, :, w * 3:w * 4] = temp[1][:, 0, :, :]

        y = self.pixel_un_shuffle(y)

        return y
