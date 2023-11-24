import math
import torch

from torch import nn


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# class CustomDataParallel(nn.DataParallel):
#     """Custom DataParallel to access the module methods."""
#
#     def __getattr__(self, key):
#         try:
#             return super().__getattr__(key)
#         except AttributeError:
#             return getattr(self.module, key)


# class RateDistortionLoss(nn.Module):
#
#     def __init__(self, lmbda=1e-2):
#         super().__init__()
#         self.mse = nn.MSELoss()
#         self.lmbda = lmbda
#
#     def forward(self, x_hat, likelihoods, target):
#         N, _, H, W = target.size()
#         out = {}
#         num_pixels = N * H * W
#
#         out["bpp_loss"] = sum(
#             (torch.log(likelihoods.item()).sum() / (-math.log(2) * num_pixels))
#         )
#         out["mse_loss"] = self.mse(x_hat, target)
#         out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
#
#         return out

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.gamma = gamma

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]

        return out


class DisDistortionLoss(nn.Module):

    def __init__(self):
        super(DisDistortionLoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, fake_gt_data_D, real_gt_data_D, targets_real, targets_fake):
        out = self.bce(fake_gt_data_D, targets_fake) + self.bce(real_gt_data_D, targets_real)

        return out
