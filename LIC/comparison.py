import math
import io
import torch
from compressai.datasets import ImageFolder
from compressai.zoo.image import _load_model
from compressai.zoo.pretrained import load_pretrained
from torch.hub import load_state_dict_from_url
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

from PIL import Image

import matplotlib.pyplot as plt

from pytorch_msssim import ms_ssim
from compressai.zoo import (bmshj2018_factorized, bmshj2018_hyperprior, mbt2018_mean, mbt2018, cheng2020_anchor)
from models.compressor import Compressor, CompressorB
from models.generator import Generator
from ipywidgets import interact, widgets

from utils.meter import AverageMeter


def read_image(filepath: str) -> torch.Tensor:
    assert filepath.is_file()
    img = Image.open(filepath).convert("RGB")
    return transforms.ToTensor()(img)


if __name__ == '__main__':
    device = 'cuda'
    metric = 'mse'  # only pre-trained model for mse are available for now
    # quality = 3
    # lower quality -> lower bit-rate (use lower quality to clearly see visual differences in the notebook)
    compressor_point = torch.load('./checkpoints/models_LIC_1e-2', map_location=device)
    ours = Compressor(N=128)
    ours.load_state_dict(compressor_point["state_dict"])
    # compressor_point1 = torch.load('./checkpoints/models_LICAtt_clic2020_1.pth.tar', map_location=device)
    # quality1 = Compressor(N=128)
    # quality1.load_state_dict(compressor_point1["state_dict"])
    #
    # print(compressor_point1['epoch'])
    # print(compressor_point1["lr_scheduler"])
    #
    # compressor_point2 = torch.load('./checkpoints/models_LICAtt_clic2020_2.pth.tar', map_location=device)
    # quality2 = Compressor(N=128)
    # quality2.load_state_dict(compressor_point2["state_dict"])
    #
    # print(compressor_point2['epoch'])
    # print(compressor_point2["lr_scheduler"])
    #
    # compressor_point3 = torch.load('./checkpoints/models_LICAtt_clic2020_3.pth.tar', map_location=device)
    # quality3 = Compressor(N=128)
    # quality3.load_state_dict(compressor_point3["state_dict"])
    #
    # print(compressor_point3['epoch'])
    # print(compressor_point3["lr_scheduler"])
    #
    # compressor_point4 = torch.load('./checkpoints/models_LICAtt_clic2020_4_192.pth.tar', map_location=device)
    # quality4 = Compressor(N=192)
    # quality4.load_state_dict(compressor_point4["state_dict"])
    # print(compressor_point4['epoch'])
    # print(compressor_point4["lr_scheduler"])
    # #
    # compressor_point5 = torch.load('./checkpoints/models_LICAtt_clic2020_5.pth.tar', map_location=device)
    # quality5 = Compressor(N=192)
    # quality5.load_state_dict(compressor_point5["state_dict"])
    #
    # print(compressor_point5['epoch'])
    # print(compressor_point5["lr_scheduler"])
    # #
    # compressor_point6 = torch.load('./checkpoints/models_LICAtt_clic2020_6.pth.tar', map_location=device)
    # quality6 = Compressor(N=192)
    # quality6.load_state_dict(compressor_point6["state_dict"])
    #
    # print(compressor_point5['epoch'])

    # root_url = "https://compressai.s3.amazonaws.com/models/v1"
    # state_dict = load_state_dict_from_url(f"{root_url}/mbt2018-mean-1-e522738d.pth.tar")
    # state_dict = load_pretrained(state_dict)
    # oursB = CompressorB()
    # oursB.load_state_dict(state_dict=state_dict)

    generator_point = torch.load('./checkpoints/models_DG_1e-2.pth', map_location=device)
    fix = Generator()
    fix.load_state_dict(generator_point['G_state_dict'])
    fix.eval().to(device)

    networks = {
        # 'bmshj2018-factorized': bmshj2018_factorized(quality=quality, pretrained=True).eval().to(device),
        # 'bmshj2018-hyperprior': bmshj2018_hyperprior(quality=quality, pretrained=True).eval().to(device),
        # 'mbt2018-mean': mbt2018_mean(quality=quality, pretrained=True).eval().to(device),
        # # 'mbt2018': mbt2018(quality=quality, pretrained=True).eval().to(device),
        'ours': ours.eval().to(device),
    }



    # img = Image.oimg = Image.open("D:\\dataset\\clic2020\\test\\img00003.png").convert('RGB')
    #
    # transform = transforms.Compose(
    #         [transforms.CenterCrop((256, 256)), transforms.ToTensor()]
    #     )
    #
    # x = transform(img).unsqueeze(0).to(device)

    # outputs = {}
    # with torch.no_grad():
    #     for name, net in networks.items():
    #         rv = net(x)
    #         rv['x_hat'].clamp_(0, 1)
    #         outputs[name] = rv
    # networks['ours'].update()
    # compressed = networks['ours'].compress(x)
    # corrupt_data, gt_data, mask, inverted_mask = networks['ours'].drop(0.25, compressed["strings"], compressed["shape"])
    # outputs['ours(dropped)'] = networks['ours'].reverse(corrupt_data)
    # fake = fix(corrupt_data)
    # fake_in = corrupt_data * inverted_mask + fake * mask
    # outputs['ours(fixed)'] = networks['ours'].reverse(fake_in)

    # reconstructions = {name: transforms.ToPILImage()(out['x_hat'].squeeze())
    #                   for name, out in outputs.items()}
    #
    #
    # diffs = [torch.mean((out['x_hat'] - x).abs(), axis=1).squeeze()
    #         for out in outputs.values()]
    #
    # fix, axes = plt.subplots((len(reconstructions) + 2) // 3, 3, figsize=(16, 12))
    # for ax in axes.ravel():
    #     ax.axis('off')
    #
    # axes.ravel()[0].imshow(transforms.ToPILImage()(x.squeeze()))
    # axes.ravel()[0].title.set_text('Original')
    #
    # for i, (name, rec) in enumerate(reconstructions.items()):
    #     axes.ravel()[i + 1].imshow(rec)  # cropped for easy comparison
    #     axes.ravel()[i + 1].title.set_text(name)
    #
    # plt.show()

    # Metric
    def compute_psnr(a, b):
        mse = torch.mean((a - b) ** 2).item()
        return -10 * math.log10(mse)


    def compute_msssim(a, b):
        return ms_ssim(a, b, data_range=1.).item()


    def compute_bpp(out_net):
        size = out_net['x_hat'].size()
        num_pixels = size[0] * size[2] * size[3]
        return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
                   for likelihoods in out_net['likelihoods'].values()).item()


    metrics = {}
    outputs = {}
    psnrs = {}
    msssims = {}
    bpps = {}
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

    with torch.no_grad():
        for name, net in networks.items():
            psnr = AverageMeter()
            msssim = AverageMeter()
            bpp = AverageMeter()
            for d in dataloader:
                d = d.to(device)
                # net.update()
                # com = net.compress(d)
                # rv = net.decompress(com['strings'], com['shape'])

                rv = net(d)
                outputs[name] = rv
                psnr.update(compute_psnr(d, rv["x_hat"]))
                msssim.update(compute_msssim(d, rv["x_hat"]))
                bpp.update(compute_bpp(rv))
            metrics[name] = {
                'psnr': psnr.avg,
                'ms-ssim': msssim.avg,
                'bit-rate': bpp.avg,
            }

    header = f'{"Model":20s} | {"PSNR [dB]"} | {"MS-SSIM":<9s} | {"Bpp":<9s}|'
    print('-' * len(header))
    print(header)
    print('-' * len(header))
    for name, m in metrics.items():
        print(f'{name:20s}', end='')
        for v in m.values():
            print(f' | {v:9.8f}', end='')
        print('|')
    print('-' * len(header))

    # fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # plt.figtext(.5, 0., '(upper-left is better)', fontsize=12, ha='center')
    # for name, m in metrics.items():
    #     axes[0].plot(m['bit-rate'], m['psnr'], 'o', label=name)
    #     axes[0].legend(loc='best')
    #     axes[0].grid()
    #     axes[0].set_ylabel('PSNR [dB]')
    #     axes[0].set_xlabel('Bit-rate [bpp]')
    #     axes[0].title.set_text('PSNR comparison')
    #
    #     axes[1].plot(m['bit-rate'], -10 * np.log10(1 - m['ms-ssim']), 'o', label=name)
    #     axes[1].legend(loc='best')
    #     axes[1].grid()
    #     axes[1].set_ylabel('MS-SSIM [dB]')
    #     axes[1].set_xlabel('Bit-rate [bpp]')
    #     axes[1].title.set_text('MS-SSIM (log) comparison')
    #
    # plt.show()



