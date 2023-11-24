import math
import random

import numpy as np
import torch
import cv2
import os
from PIL import Image
import torchvision
from compressai.datasets import ImageFolder
from pytorch_msssim import ms_ssim
from torch import nn
from torch.utils.data import DataLoader

from models.compressor import Compressor
from models.generator import Generator
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from torchvision.utils import save_image
from torchvision import transforms

def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return -10 * math.log10(mse)


def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()

if __name__ == '__main__':
    device = "cuda"
    train_transforms = transforms.Compose(
        [transforms.RandomCrop((256, 256)), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop((256, 256)), transforms.ToTensor()]
    )

    transform = transforms.Compose(
        [transforms.ToTensor()]
    )

    net = Compressor()
    checkpoint = torch.load('./checkpoints/models_LIC_1e-2.pth.tar', map_location=device)
    net.load_state_dict(checkpoint["state_dict"])
    net.to(device)
    net.eval()

    fix = Generator()
    savepoint = torch.load('./checkpoints/models_DG_1e-2.pth', map_location=device)
    fix.load_state_dict(savepoint['G_state_dict'])
    fix.to(device)
    fix.eval()

    mse = nn.MSELoss(reduction='mean')
    mse.to(device)

    img = Image.open("dataset/Kodak24/test/img19.png").convert('RGB')
    img_in = transform(img).unsqueeze(0).to(device)
    save_image(img_in[0], 'out/imgin.png', padding=0)

    with torch.no_grad():
        net.update()
        output = net.compress(img_in)
        out = net.decompress(output["strings"], output["shape"])
    img_out = out["x_hat"][0]
    save_image(img_out, 'out/imgout_1.png', padding=0)
    print(compute_psnr(img_in, out["x_hat"]))
    print(compute_msssim(img_in, out["x_hat"]))

    with torch.no_grad():
        net.update()
        output = net.compress(img_in)
        corrupt_data, gt_data, mask, inverted_mask, num = net.drop(0.084, output["strings"], output["shape"])
        print(num)
        gauss = torch.normal(0, 0.2, corrupt_data.shape).to(device)
        out = net.reverse(corrupt_data * inverted_mask + gauss * mask)

        img_out = out["x_hat"][0]
        save_image(img_out, 'out/imgout_2.png', padding=0)
        print(compute_psnr(img_in, out["x_hat"]))
        print(compute_msssim(img_in, out["x_hat"]))

        fake = fix(corrupt_data)
        fake_in = corrupt_data * inverted_mask + fake * mask

        out = net.reverse(fake_in)

        img_out = out["x_hat"][0]
        save_image(img_out, 'out/imgout_3.png', padding=0)
        print(compute_psnr(img_in, out["x_hat"]))
        print(compute_msssim(img_in, out["x_hat"]))
