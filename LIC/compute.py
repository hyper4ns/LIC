import math

import torch

from PIL import Image
from compressai.datasets import ImageFolder
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch_msssim import ms_ssim
from models.compressor import Compressor, CompressorB
from models.generator import Generator
from utils.meter import AverageMeter


def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return -10 * math.log10(mse)


def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()

if __name__ == '__main__':
    device = 'cuda'

    compressor_point = torch.load('./checkpoints/models_LIC_1e-2.pth.tar', map_location=device)
    icn = Compressor(N=192)
    icn.load_state_dict(compressor_point["state_dict"])
    icn.eval().to(device)

    generator_point = torch.load('./checkpoints/models_DG_1e-2.pth', map_location=device)
    fpn = Generator()
    fpn.load_state_dict(generator_point['G_state_dict'])
    fpn.eval().to(device)

    transform = transforms.Compose(
        [transforms.ToTensor()]
    )

    test_dataset = ImageFolder("dataset/Kodak24", split="test", transform=transform)
    dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    psnrs = {}
    msssims = {}

    with torch.no_grad():
        psnr1 = AverageMeter()
        msssim1 = AverageMeter()
        psnr2 = AverageMeter()
        msssim2 = AverageMeter()
        psnr3 = AverageMeter()
        msssim3 = AverageMeter()
        for d in dataloader:
            d = d.to(device)
            icn.update()
            output = icn.compress(d)

            com1 = icn.decompress(output['strings'], output['shape'])
            psnr1.update(compute_psnr(d, com1["x_hat"]))
            msssim1.update(compute_msssim(d, com1["x_hat"]))

            corrupt_data, gt_data, mask, inverted_mask = icn.drop(0.25, output['strings'], output['shape'])
            com2 = icn.reverse(corrupt_data)

            psnr2.update(compute_psnr(d, com2["x_hat"]))
            msssim2.update(compute_msssim(d, com2["x_hat"]))

            fake = fpn(corrupt_data)
            fix = corrupt_data * inverted_mask + fake * mask
            com3 = icn.reverse(fix)

            psnr3.update(compute_psnr(d, com3["x_hat"]))
            msssim3.update(compute_msssim(d, com3["x_hat"]))

        print(psnr1.avg)
        print(psnr2.avg)
        print(psnr3.avg)
        print(psnr3.avg / psnr2.avg)
        print(msssim1.avg)
        print(msssim2.avg)
        print(msssim3.avg)
        print(msssim3.avg / msssim2.avg)




    # with torch.no_grad():
    #     count = 0
    #     avlen = 0
    #     avSiz = 0
    #     num = 0
    #     for d in dataloader:
    #         count += 1
    #         d = d.to(device)
    #         icn.update()
    #         com = icn.compress(d)
    #         y_strings = com['strings'][0]
    #         n = len(y_strings)
    #         sum = 0
    #         for s in y_strings:
    #             sum += len(s[0])
    #         avg = sum/n
    #
    #         avlen += avg
    #         avSiz += sum
    #         num = n
    #     avlen /= count
    #     avSiz /= count
    #     print(n)
    #     print(avSiz)
    #     print(avlen)
