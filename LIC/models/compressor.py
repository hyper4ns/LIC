import math
import random

import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import MeanScaleHyperprior

from models import layers

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class Compressor(nn.Module):
    r"""
        Layers:
            g_a: general_analysis
            g_s: general_synthesis
            h_a: hyper_analysis
            h_s: hyper_synthesis
            p_a: pack_analysis
            p_s: pack_synthesis
        Args:
            N (int): Number of analysis-channels
            M (int): Number of hyper-analysis-channels in the expansion layers (last layer of the
                encoder and last layer of the hyperprior decoder)
        """

    def __init__(self, N=192):
        super(Compressor, self).__init__()

        self.g_a = layers.GeneralAnalysisLayer(N=N)
        self.g_s = layers.GeneralSynthesisLayer(N=N)
        self.h_a = layers.HyperAnalysisLayer(N=N)
        self.h_s = layers.HyperSynthesisLayer(N=N)
        self.p_a = layers.PackAnalysisLayer()
        self.p_s = layers.PackSynthesisLayer()
        self.em_z = EntropyBottleneck(channels=N)
        self.em_y = GaussianConditional(None)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        y_packs = self.p_a(y)

        z_hat, z_likelihoods = self.em_z(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        scale_packs = self.p_a(scales_hat)
        mean_packs = self.p_a(means_hat)

        y_pack_hats = []
        y_pack_likelihoods = []
        for i, y_pack in enumerate(y_packs):
            y_pack_hat, y_pack_likelihood = self.em_y(y_pack, scale_packs[i], means=mean_packs[i])
            y_pack_hats.append(y_pack_hat)
            y_pack_likelihoods.append(y_pack_likelihood)
        y_likelihoods = self.p_s(y_pack_likelihoods)

        y_hat = self.p_s(y_pack_hats)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def update(self, scale_table=None, force=False):
        e_updated = self.em_z.update(force=force)
        if scale_table is None:
            scale_table = get_scale_table()
        g_updated = self.em_y.update_scale_table(scale_table, force=force)
        updated = e_updated and g_updated
        return updated

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        y_packs = self.p_a(y)

        z_strings = self.em_z.compress(z)
        z_hat = self.em_z.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.em_y.build_indexes(scales_hat)

        indexes_packs = self.p_a(indexes)
        mean_packs = self.p_a(means_hat)

        y_strings = []
        for i, y_pack in enumerate(y_packs):
            y_string = self.em_y.compress(y_pack, indexes_packs[i], means=mean_packs[i])
            y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.em_z.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.em_y.build_indexes(scales_hat)
        indexes_packs = self.p_a(indexes)
        mean_packs = self.p_a(means_hat)

        y_packs_hat = []
        for i, y_string in enumerate(strings[0]):
            y_hat_pack = self.em_y.decompress(
                y_string, indexes_packs[i], means=mean_packs[i]
            )
            y_packs_hat.append(y_hat_pack)

        y_hat = self.p_s(y_packs_hat)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def drop(self, rate, strings, shape):
        drop_packs = []
        for i, y in enumerate(strings[0]):
            # if (i + 2) % 4 == 0:
            #     drop_packs.append(i)
            a = random.random()
            if a < rate:
                drop_packs.append(i)

        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.em_z.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.em_y.build_indexes(scales_hat)
        indexes_packs = self.p_a(indexes)
        mean_packs = self.p_a(means_hat)

        y_packs_hat = []
        mask_packs = []
        for i, y_string in enumerate(strings[0]):
            y_hat_pack = self.em_y.decompress(
                y_string, indexes_packs[i], means=mean_packs[i]
            )

            if drop_packs.count(i) > 0:
                mask_pack = torch.ones(indexes_packs[i].size()).to("cuda")
            else:
                mask_pack = torch.zeros(indexes_packs[i].size()).to("cuda")

            y_packs_hat.append(y_hat_pack)
            mask_packs.append(mask_pack)

        gt_data = self.p_s(y_packs_hat)
        mask = self.p_s(mask_packs)
        inverted_mask = ((mask * -1) + 1).to("cuda")
        corrupt_data = gt_data * inverted_mask

        return corrupt_data, gt_data, mask, inverted_mask, len(drop_packs)

    def reverse(self, y_hat):
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


class CompressorB(MeanScaleHyperprior):
    def __init__(self):
        super(CompressorB, self).__init__(N=128, M=192)
        self.p_a = layers.PackAnalysisLayer()
        self.p_s = layers.PackSynthesisLayer()

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        # y_packs = self.p_a(y)

        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        # scale_packs = self.p_a(scales_hat)
        # mean_packs = self.p_a(means_hat)
        #
        # y_pack_hats = []
        # y_pack_likelihoods = []
        # for i, y_pack in enumerate(y_packs):
        #     y_pack_hat, y_pack_likelihood = self.gaussian_conditional(y_pack, scale_packs[i], means=mean_packs[i])
        #     y_pack_hats.append(y_pack_hat)
        #     y_pack_likelihoods.append(y_pack_likelihood)
        # y_likelihoods = self.p_s(y_pack_likelihoods)
        #
        # y_hat = self.p_s(y_pack_hats)

        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        y_packs = self.p_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)

        indexes_packs = self.p_a(indexes)
        mean_packs = self.p_a(means_hat)

        y_strings = []
        for i, y_pack in enumerate(y_packs):
            y_string = self.gaussian_conditional.compress(y_pack, indexes_packs[i], means=mean_packs[i])
            y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        indexes_packs = self.p_a(indexes)
        mean_packs = self.p_a(means_hat)

        y_packs_hat = []
        for i, y_string in enumerate(strings[0]):
            y_hat_pack = self.gaussian_conditional.decompress(
                y_string, indexes_packs[i], means=mean_packs[i]
            )
            y_packs_hat.append(y_hat_pack)

        y_hat = self.p_s(y_packs_hat)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


def build_model():
    input_image = torch.rand([1, 3, 256, 256])
    input_image = (input_image * 100)
    net = Compressor()
    out = net(input_image)
    net.update()
    q = net.compress(input_image)
    out = net.decompress(q["strings"], q["shape"])


if __name__ == '__main__':
    build_model()
