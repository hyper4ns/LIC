import random

import numpy as np
import torch
import os
from models.pixeltrans import PackAnalysisLayer, PackSynthesisLayer


class FeatureHandler:
    def __init__(self, features_dir, drop):
        self.features_dir = features_dir
        self.drop = drop
        self.pack = PackAnalysisLayer(4)
        self.unpack = PackSynthesisLayer(4)

    def load_features(self, dir):
        features = torch.load(dir)
        return features

    def get_batch(self, device):
        features = self.load_features(self.features_dir).to(device)


        # pack mask
        mask = torch.zeros(features.shape).to(device)
        gauss = torch.normal(0, 0.2, features.shape).to(device)
        packs = self.pack(mask)
        for i, pack in enumerate(packs):
            a = random.random()
            if a < self.drop:
                pack[:, :, :, :] = 1
        mask = self.unpack(packs)
        inverted_masks = ((mask * -1) + 1).to(device)
        corrupt_features = (features * inverted_masks + gauss * mask).to(device)
        # corrupt_features = (features * inverted_masks).to(device)

        return corrupt_features, features, mask, inverted_masks
