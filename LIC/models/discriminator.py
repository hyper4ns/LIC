import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.block1 = torch.nn.Sequential()
        self.block1.add_module("conv_1", torch.nn.Conv2d(192, 192, kernel_size=3, padding=1))
        self.block1.add_module("bn_1", torch.nn.BatchNorm2d(192))
        self.block1.add_module("relu_1", torch.nn.ReLU())

        self.block2 = torch.nn.Sequential()
        self.block2.add_module("conv_2", torch.nn.Conv2d(192, 256, kernel_size=3, padding=1))
        self.block2.add_module("bn_2", torch.nn.BatchNorm2d(256))
        self.block2.add_module("relu_2", torch.nn.ReLU())

        self.block3 = torch.nn.Sequential()
        self.block3.add_module("conv_3", torch.nn.Conv2d(256, 256, kernel_size=3, padding=1))
        self.block3.add_module("bn_3", torch.nn.BatchNorm2d(256))
        self.block3.add_module("relu_3", torch.nn.ReLU())
        self.block3.add_module("maxpool_3", torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.block4 = torch.nn.Sequential()
        self.block4.add_module("conv_4", torch.nn.Conv2d(256, 512, kernel_size=3, padding=1))
        self.block4.add_module("bn_4", torch.nn.BatchNorm2d(512))
        self.block4.add_module("relu_4", torch.nn.ReLU())
        self.block4.add_module("maxpool_4", torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(8192, 4096)
        self.fc2 = nn.Linear(4096, 1)
        # self.fc3 = nn.Linear(2048, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.block1.forward(x)
        # print(x.shape)
        x = self.block2.forward(x)
        # print(x.shape)
        x = self.block3.forward(x)
        # print(x.shape)
        x = self.block4.forward(x)
        # print(x.shape)
        x = self.flat(x)
        # print(x.shape)
        x = self.relu(self.fc1.forward(x))
        # x = self.relu(self.fc2.forward(x))
        x = self.sigmoid.forward(self.fc2.forward(x))
        return x


def build_model():
    input_image = torch.rand([4, 192, 16, 16])
    input_image = input_image * 100
    net = Discriminator()
    feature = net(input_image)
    print(feature.size())


if __name__ == '__main__':
    build_model()
