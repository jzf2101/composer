# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Stochastic forward functions for ResNet Bottleneck modules."""

import torch
from torchvision.models.resnet import Bottleneck


def block_stochastic_forward(self, x):
    """ResNet Bottleneck forward function where the layers are randomly
        skipped with probability ``drop_rate`` during training.
    """

    identity = x

    sample = (not self.training) or bool(torch.bernoulli(1 - self.drop_rate))

    if sample:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if not self.training:
            out = out * (1 - self.drop_rate)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
    else:
        if self.downsample is not None:
            out = self.relu(self.downsample(identity))
        else:
            out = identity
    return out


def _sample_drop(x: torch.Tensor, sample_drop_rate: float, is_training: bool):
    """Randomly drops samples from the input batch according to the `sample_drop_rate`.

    This is implemented by setting the samples to be dropped to zeros.
    """

    keep_probability = (1 - sample_drop_rate)
    if not is_training:
        return x * keep_probability
    rand_dim = [x.shape[0]] + [1] * len(x.shape[1:])
    sample_mask = keep_probability + torch.rand(rand_dim, dtype=x.dtype, device=x.device)
    sample_mask.floor_()  # binarize
    x *= sample_mask
    return x


def sample_stochastic_forward(self, x):
    """ResNet Bottleneck forward function where samples are randomly
        dropped with probability ``drop_rate`` during training.
    """

    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    if self.drop_rate:
        out = _sample_drop(out, self.drop_rate, self.training)
    out += identity

    return self.relu(out)


def make_resnet_bottleneck_stochastic(module: Bottleneck, module_index: int, module_count: int, drop_rate: float,
                                      drop_distribution: str, stochastic_method: str):
    """Model surgery policy that dictates how to convert a ResNet Bottleneck layer into a stochastic version.
    """

    if drop_distribution == 'linear':
        drop_rate = ((module_index + 1) / module_count) * drop_rate
    module.drop_rate = torch.tensor(drop_rate)

    stochastic_func = block_stochastic_forward if stochastic_method == 'block' else sample_stochastic_forward
    module.forward = stochastic_func.__get__(module)  # Bind new forward function to ResNet Bottleneck Module

    return module
