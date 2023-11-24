import torch
import torch.nn as nn
from compressai.layers import ResidualBlock, ResidualBlockWithStride, ResidualBlockUpsample, conv3x3, subpel_conv3x3


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.block1 = nn.Sequential(
            ResidualBlockWithStride(192, 256, stride=1),
            ResidualBlock(256, 256),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.block5 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.block6 = nn.Sequential(
            ResidualBlock(256, 256),
            ResidualBlockUpsample(256, 192, 1)
        )

    def forward(self, x):
        x = self.block1.forward(x)
        # print(x.shape)
        x = self.block2.forward(x)
        # print(x.shape)
        x = self.block3.forward(x)
        # print(x.shape)
        x = self.block4.forward(x)
        # print(x.shape)
        x = self.block5.forward(x)
        # print(x.shape)
        x = self.block6.forward(x)
        return x


def build_model():
    input_image = torch.rand([4, 192, 16, 16])
    input_image = input_image * 100
    net = Generator()
    feature = net(input_image)


if __name__ == '__main__':
    build_model()


