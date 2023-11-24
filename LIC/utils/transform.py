import torch
import torch.nn as nn


def transform(latent):
    n, c, h, w = latent.shape()
    ps = nn.PixelShuffle(2)
